import numpy as np

import torch
import torch.optim as optim

from mushroom.algorithms.agent import Agent
from mushroom.policy import Policy
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import TorchApproximator
from mushroom.utils.replay_memory import ReplayMemory
from mushroom.utils.value_functions import compute_advantage
from mushroom.utils.dataset import parse_dataset
from mushroom.utils.torch import to_float_tensor

from copy import deepcopy
from itertools import chain


class DeepAC(Agent):
    """
    Base class for algorithms that uses the reparametrization trick, such as
    SAC, DDPG and TD3.

    """
    def __init__(self, policy, mdp_info, actor_optimizer, parameters):
        if actor_optimizer is not None:
            self._optimizer = actor_optimizer['class'](
                parameters, **actor_optimizer['params']
            )

        super().__init__(policy, mdp_info)

    def _optimize_actor_parameters(self, loss):
        """
        Method used to update actor parameters to maximize a given loss.

        Args:
            loss (torch.tensor): the loss computed by the algorithm.

        """
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()


class A2C(DeepAC):
    """
    Advantage Actor Critic algorithm (A2C).
    Synchronous version of the A3C algorithm.
    "Asynchronous Methods for Deep Reinforcement Learning".
    Mnih V. et. al.. 2016.

    """
    def __init__(self, mdp_info, policy, critic_params, actor_optimizer,
                 ent_coeff, critic_fit_params=None):
        """
        Constructor.

        Args:
            policy (TorchPolicy): torch policy to be learned by the algorithm
            critic_params (dict): parameters of the critic approximator to
                build;
            actor_optimizer (dict): parameters to specify the actor optimizer
                algorithm;
            ent_coeff (float, 0): coefficient for the entropy penalty;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.

        """
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._entropy_coeff = ent_coeff

        self._V = Regressor(TorchApproximator, **critic_params)

        super().__init__(policy, mdp_info, actor_optimizer, policy.parameters())

    def fit(self, dataset):
            state, action, reward, next_state, absorbing, _ = parse_dataset(dataset)

            adv, v = compute_advantage(self._V, state, next_state,
                                       reward, absorbing, self.mdp_info.gamma)
            self._V.fit(state, v, **self._critic_fit_params)

            loss = self._loss(state, action, adv)
            self._optimize_actor_parameters(loss)

    def _loss(self, state, action, adv):
        use_cuda = self.policy.use_cuda

        s = to_float_tensor(state, use_cuda)
        a = to_float_tensor(action, use_cuda)

        adv_t = to_float_tensor(adv, use_cuda)

        gradient_loss = -torch.mean(self.policy.log_prob_t(s, a)*adv_t)
        entropy_loss = -self.policy.entropy_t(s)

        return gradient_loss + self._entropy_coeff * entropy_loss


class DDPG(DeepAC):
    """
    Deep Deterministic Policy Gradient algorithm.
    "Continuous Control with Deep Reinforcement Learning".
    Lillicrap T. P. et al.. 2016.

    """
    def __init__(self, mdp_info, policy_class, policy_params,
                 batch_size, initial_replay_size, max_replay_size,
                 tau, critic_params, actor_params, actor_optimizer,
                 policy_delay=1, critic_fit_params=None):
        """
        Constructor.

        Args:
            policy_class (Policy): class of the policy;
            policy_params (dict): parameters of the policy to build;
            batch_size (int): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            tau (float): value of coefficient for soft updates;
            actor_params (dict): parameters of the actor approximator to
                build;
            critic_params (dict): parameters of the critic approximator to
                build;
            actor_optimizer (dict): parameters to specify the actor optimizer
                algorithm;
            policy_delay (int, 1): the number of updates of the critic after
                which an actor update is implemented;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator;

        """

        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = batch_size
        self._tau = tau
        self._policy_delay = policy_delay
        self._fit_count = 0

        self._replay_memory = ReplayMemory(initial_replay_size, max_replay_size)

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator,
                                                     **target_critic_params)

        target_actor_params = deepcopy(actor_params)
        self._actor_approximator = Regressor(TorchApproximator,
                                             **actor_params)
        self._target_actor_approximator = Regressor(TorchApproximator,
                                                    **target_actor_params)

        self._init_target()

        policy = policy_class(self._actor_approximator, **policy_params)

        policy_parameters = self._actor_approximator.model.network.parameters()

        super().__init__(policy, mdp_info, actor_optimizer, policy_parameters)

    def fit(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ =\
                self._replay_memory.get(self._batch_size)

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            self._critic_approximator.fit(state, action, q,
                                          **self._critic_fit_params)

            if self._fit_count % self._policy_delay == 0:
                loss = self._loss(state)
                self._optimize_actor_parameters(loss)

            self._update_target()

            self._fit_count += 1

    def _loss(self, state):
        action = self._actor_approximator(state, output_tensor=True)
        q = self._critic_approximator(state, action, output_tensor=True)

        return -q.mean()

    def _init_target(self):
        """
        Init weights for target approximators

        """
        self._target_actor_approximator.set_weights(
            self._actor_approximator.get_weights())
        self._target_critic_approximator.set_weights(
            self._critic_approximator.get_weights())

    def _update_target(self):
        """
        Update the target networks.

        """
        critic_weights = self._tau * self._critic_approximator.get_weights()
        critic_weights += (1 - self._tau) * self._target_critic_approximator.get_weights()
        self._target_critic_approximator.set_weights(critic_weights)

        actor_weights = self._tau * self._actor_approximator.get_weights()
        actor_weights += (1 - self._tau) * self._target_actor_approximator.get_weights()
        self._target_actor_approximator.set_weights(actor_weights)

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
        a = self._target_actor_approximator(next_state)

        q = self._target_critic_approximator.predict(next_state, a)
        q *= 1 - absorbing

        return q


class TD3(DDPG):
    """
    Twin Delayed DDPG algorithm.
    "Addressing Function Approximation Error in Actor-Critic Methods".
    Fujimoto S. et al.. 2018.

    """
    def __init__(self, mdp_info, policy_class, policy_params,
                 batch_size, initial_replay_size, max_replay_size,
                 tau, critic_params, actor_params, actor_optimizer,
                 policy_delay=2, noise_std=.2, noise_clip=.5,
                 critic_fit_params=None):
        """
        Constructor.

        Args:
            policy_class (Policy): class of the policy;
            policy_params (dict): parameters of the policy to build;
            batch_size (int): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            tau (float): value of coefficient for soft updates;
            critic_params (dict): parameters of the critic approximator to
                build;
            actor_params (dict): parameters of the actor approximator to
                build;
            actor_optimizer (dict): parameters to specify the actor
                optimizer algorithm;
            policy_delay (int, 2): the number of updates of the critic after
                which an actor update is implemented;
            noise_std (float, .2): standard deviation of the noise used for
                policy smoothing;
            noise_clip (float, .5): maximum absolute value for policy smoothing
                noise;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.

        """
        self._noise_std = noise_std
        self._noise_clip = noise_clip

        if 'n_models' in critic_params.keys():
            assert(critic_params['n_models'] >= 2)
        else:
            critic_params['n_models'] = 2

        if 'prediction' not in critic_params.keys():
            critic_params['prediction'] = 'min'

        super().__init__(mdp_info, policy_class, policy_params,
                         batch_size, initial_replay_size, max_replay_size,
                         tau, critic_params, actor_params, actor_optimizer,
                         policy_delay, critic_fit_params)

    def _loss(self, state):
        action = self._actor_approximator(state, output_tensor=True)
        q = self._critic_approximator(state, action, idx=0, output_tensor=True)

        return -q.mean()

    def _init_target(self):
        """
        Initialize weights for target approximators.

        """
        self._target_actor_approximator.set_weights(
            self._actor_approximator.get_weights())
        for i in range(len(self._critic_approximator)):
            self._target_critic_approximator.model[i].set_weights(
                self._critic_approximator.model[i].get_weights())

    def _update_target(self):
        """
        Update the target networks.

        """
        for i in range(len(self._target_critic_approximator)):
            critic_weights_i = self._tau * self._critic_approximator.model[i].get_weights()
            critic_weights_i += (1 - self._tau) * self._target_critic_approximator.model[i].get_weights()
            self._target_critic_approximator.model[i].set_weights(critic_weights_i)

        actor_weights = self._tau * self._actor_approximator.get_weights()
        actor_weights += (1 - self._tau) * self._target_actor_approximator.get_weights()
        self._target_actor_approximator.set_weights(actor_weights)

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
        a = self._target_actor_approximator(next_state)

        low = self.mdp_info.action_space.low
        high = self.mdp_info.action_space.high
        eps = np.random.normal(scale=self._noise_std, size=a.shape)
        eps_clipped = np.clip(eps, -self._noise_clip, self._noise_clip)
        a_smoothed = np.clip(a + eps_clipped, low, high)

        q = self._target_critic_approximator.predict(next_state, a_smoothed)
        q *= 1 - absorbing

        return q


class SACPolicy(Policy):
    """
    Class used to implement the policy used by the Soft Actor-Critic
    algorithm. The policy is a Gaussian policy squashed by a tanh.
    This class implements the compute_action_and_log_prob and the
    compute_action_and_log_prob_t methods, that are fundamental for
    the internals calculations of the SAC algorithm.

    """
    def __init__(self, mu_approximator, sigma_approximator, min_a, max_a):
        """
        Constructor.

        Args:
            mu_approximator (Regressor): a regressor computing mean in given a
                state;
            sigma_approximator (Regressor): a regressor computing the variance
                in given a state;
            min_a (np.ndarray): a vector specifying the minimum action value
                for each component;
            max_a (np.ndarray): a vector specifying the maximum action value
                for each component.

        """
        self._mu_approximator = mu_approximator
        self._sigma_approximator = sigma_approximator
        self._delta_a = torch.from_numpy(.5 * (max_a - min_a))
        self._central_a = torch.from_numpy(.5 * (max_a + min_a))

    def __call__(self, state, action):
        raise NotImplementedError

    def draw_action(self, state):
        return self.compute_action_and_log_prob_t(
            state, compute_log_prob=False).detach().cpu().numpy()

    def compute_action_and_log_prob(self, state):
        """
        Function that samples actions using the reparametrization trick and
        the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled.

        Returns:
            The actions sampled and the log probability as numpy arrays.

        """
        a, log_prob = self.compute_action_and_log_prob_t(state)
        return a.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def compute_action_and_log_prob_t(self, state, compute_log_prob=True):
        """
        Function that samples actions using the reparametrization trick and,
        optionally, the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled;
            compute_log_prob (bool, True): whether to compute the log
            probability or not.

        Returns:
            The actions sampled and, optionally, the log probability as torch
            tensors.

        """
        dist = self.distribution(state)
        a_raw = dist.rsample()
        a = torch.tanh(a_raw)
        a_true = a * self._delta_a + self._central_a

        if compute_log_prob:
            log_prob = dist.log_prob(a_raw).flatten()
            log_prob -= torch.log(1. - a.pow(2) + 1e-6).sum(dim=1)
            return a_true, log_prob
        else:
            return a_true

    def distribution(self, state):
        mu = self._mu_approximator.predict(state, output_tensor=True)
        log_sigma = self._sigma_approximator.predict(state, output_tensor=True)
        return torch.distributions.Normal(mu, log_sigma.exp())

    def reset(self):
        pass


class SAC(DeepAC):
    """
    Soft Actor-Critic algorithm.
    "Soft Actor-Critic Algorithms and Applications".
    Haarnoja T. et al.. 2019.

    """
    def __init__(self, mdp_info,
                 batch_size, initial_replay_size, max_replay_size,
                 warmup_transitions, tau, lr_alpha,
                 actor_mu_params, actor_sigma_params,
                 actor_optimizer, critic_params,
                 target_entropy=None, critic_fit_params=None):
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
            actor_sigma_params (dict): parameters of the actor sigm
                approximator to build;
            actor_optimizer (dict): parameters to specify the actor
                optimizer algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            target_entropy (float, None): target entropy for the policy, if
                None a default value is computed ;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.

        """
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = batch_size
        self._warmup_transitions = warmup_transitions
        self._tau = tau

        if target_entropy is None:
            self._target_entropy = -np.prod(mdp_info.action_space.shape).astype(np.float32)
        else:
            self._target_entropy = target_entropy

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
        self._alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)

        actor_mu_approximator = Regressor(TorchApproximator,
                                          **actor_mu_params)
        actor_sigma_approximator = Regressor(TorchApproximator,
                                             **actor_sigma_params)

        policy = SACPolicy(actor_mu_approximator,
                           actor_sigma_approximator,
                           mdp_info.action_space.low,
                           mdp_info.action_space.high)

        self._init_target()

        policy_parameters = chain(actor_mu_approximator.model.network.parameters(),
                                  actor_sigma_approximator.model.network.parameters())
        super().__init__(policy, mdp_info, actor_optimizer, policy_parameters)

    def fit(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(self._batch_size)

            if self._replay_memory.size > self._warmup_transitions:
                action_new, log_prob = self.policy.compute_action_and_log_prob_t(state)
                loss = self._loss(state, action_new, log_prob)
                self._optimize_actor_parameters(loss)
                self._update_alpha(log_prob.detach())

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            self._critic_approximator.fit(state, action, q,
                                          **self._critic_fit_params)

            self._update_target()

    def _init_target(self):
        """
        Init weights for target approximators.

        """
        for i in range(len(self._critic_approximator)):
            self._target_critic_approximator.model[i].set_weights(
                self._critic_approximator.model[i].get_weights())

    def _loss(self, state, action_new, log_prob):
        q_0 = self._critic_approximator(state, action_new,
                                        output_tensor=True, idx=0)
        q_1 = self._critic_approximator(state, action_new,
                                        output_tensor=True, idx=1)

        q = torch.min(q_0, q_1)

        return (self._alpha * log_prob - q).mean()

    def _update_alpha(self, log_prob):
        alpha_loss = - (self._log_alpha * (log_prob + self._target_entropy)).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

    def _update_target(self):
        """
        Update the target networks.

        """
        for i in range(len(self._target_critic_approximator)):
            critic_weights_i = self._tau * self._critic_approximator.model[i].get_weights()
            critic_weights_i += (1 - self._tau) * self._target_critic_approximator.model[i].get_weights()
            self._target_critic_approximator.model[i].set_weights(critic_weights_i)

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
        a, log_prob_next = self.policy.compute_action_and_log_prob(next_state)

        q = self._target_critic_approximator.predict(next_state, a) - self._alpha_np * log_prob_next
        q *= 1 - absorbing

        return q

    @property
    def _alpha(self):
        return self._log_alpha.exp()

    @property
    def _alpha_np(self):
        return self._alpha.detach().cpu().numpy()
