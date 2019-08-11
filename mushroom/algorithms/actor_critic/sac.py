from copy import deepcopy

import numpy as np

import torch
import torch.optim as optim

from mushroom.algorithms.actor_critic import ReparametrizationAC
from mushroom.policy import Policy
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import TorchApproximator
from mushroom.utils.replay_memory import ReplayMemory


class SACGaussianPolicy(Policy):
    def __init__(self, mu_approximator, sigma_approximator):
        """
        Constructor.

        Args:
            approximator (Regressor): a regressor computing mean and variance given a state
        """
        self._mu_approximator = mu_approximator
        self._sigma_approximator = sigma_approximator

    def __call__(self, state, action, use_log=True):
        if use_log:
            return self._mu_approximator.log_prob(state, action)
        else:
            return np.exp(self._mu_approximator.log_prob(state, action))

    def draw_action(self, state):
        mu, sigma = self._approximator.predict(state)

        a = mu + sigma*np.random.randn(len(sigma))
        return a

    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    def reset(self):
        pass


class SAC(ReparametrizationAC):
    """
    Soft Actor-Critic algorithm.
    "Soft Actor-Critic Algorithms and Applications".
    Haarnoja T. et al.. 2019
    """
    def __init__(self, mdp_info,
                 batch_size, initial_replay_size, max_replay_size,
                 warmup_transitions, tau, lr_alpha,
                 actor_mu_params, actor_sigma_params,
                 actor_optimizer, critic_params, critic_fit_params=None):
        """
        Constructor.

        Args:
            batch_size (int): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            warmup_transitions (int): number of samples to accumulate in the
                replay memory to start the policy fitting;
            tau (float): value of coefficient for soft updates;
            lr_alpha (float): Learning rate for the entropy coefficient;
            actor_mu_params (dict): parameters of the actor mean approximator
                to build;
            actor_sigma_params (dict): parameters of the actor sigma approximator
                to build;
            critic_params (dict): parameters of the critic approximator to
                build;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator;
        """
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = batch_size
        self._warmup_transitions = warmup_transitions
        self._tau = tau
        self._target_entropy = - mdp_info.action_space.shape[0]

        self._replay_memory = ReplayMemory(initial_replay_size, max_replay_size)

        if 'n_models' in critic_params.keys():
            assert critic_params['n_models'] == 2
        else:
            critic_params['n_models'] = 2

        if 'prediction' in critic_params.keys():
            assert critic_params['prediction'] == 'min'
        else:
            critic_params['prediction'] = 'min'

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator,
                                                     **target_critic_params)

        self._log_alpha = torch.tensor(0., requires_grad=True, dtype=torch.float32)
        self._alpha_optim = optim.Adam([self.log_alpha], lr=lr_alpha)

        self._actor_mu_approximator = Regressor(TorchApproximator,
                                                **actor_mu_params)
        self._actor_sigma_approximator = Regressor(TorchApproximator,
                                                   **actor_sigma_params)
        policy = SACGaussianPolicy(self._actor_mu_approximator,
                                   self._actor_sigma_approximator)

        self._init_target()

        policy_parameters = self._actor_mu_approximator.network.params() + \
                            self._actor_sigma_approximator.network.params()
        super().__init__(policy, mdp_info, actor_optimizer, policy_parameters)

    def fit(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(self._batch_size)

            if self._replay_memory.size() > self._warmup_transitions:
                eps = np.random.randn(action.shape)
                loss = self._loss(action, state, eps)
                self._optimize_actor_parameters(loss)
                self._update_alpha(state, eps)

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next
            self._critic_approximator.fit(state, action, q,
                                          **self._critic_fit_params)

            self._update_target()

    def _init_target(self):
        """
        Init weights for target approximators

        """
        for i in range(len(self._critic_approximator)):
            self._target_critic_approximator.model[i].set_weights(
                self._critic_approximator.model[i].get_weights())

    def _loss(self, action, state, eps):
        mu = self._actor_mu_approximator.predict(state)
        sigma = self._actor_sigma_approximator.predict()

        action_new = mu + eps * sigma

        q_0 = self._critic_approximator(state, action_new,
                                        tensor_output=True, idx=0)
        q_1 = self._critic_approximator(state, action_new,
                                        tensor_output=True, idx=1)

        q = torch.min(q_0, q_1)

        log_prob = self._log_prob(state, action)

        return (self._alpha() * log_prob - q).mean()

    def _update_target(self):
        """
        Update the target networks.

        """
        for i in range(len(self._target_critic_approximator)):
            critic_weights_i = self._tau * self._critic_approximator.model[i].get_weights()
            critic_weights_i += (1 - self._tau) * self._target_critic_approximator.model[i].get_weights()
            self._target_critic_approximator.model[i].set_weights(critic_weights_i)

    def _compute_log_prob(self, state, action):
        return self._critic_approximator.model.network(state, action)

    def _alpha(self):
        return self.log_alpha.exp()

    def _alpha_np(self):
        return self._alpha().detach().cpu().numpy()

    def _update_alpha(self, state, eps):
        log_prob = self._compute_log_prob(state, eps)
        alpha_loss = - (self._log_alpha * (log_prob + self._target_entropy)).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.

        Returns:
            Action-values returned by the critic for ``next_state`` and the
            action returned by the actor.

        """
        a = self.policy(next_state)
        log_prob_next = self.policy(next_state, a)

        q = self._target_critic_approximator.predict(next_state, a) - self._alpha_np() * log_prob_next
        q *= 1 - absorbing

        return q