import numpy as np

import torch
import torch.optim as optim

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.replay_memory import ReplayMemory

from copy import deepcopy


class SAC(DeepAC):
    """
    Soft Actor-Critic algorithm.
    "Soft Actor-Critic Algorithms and Applications".
    Haarnoja T. et al.. 2019.

    """
    def __init__(self, mdp_info, policy, actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, warmup_transitions, tau,
                 lr_alpha, target_entropy=None, critic_fit_params=None):
        """
        Constructor.

        Args:
            policy (TorchPolicy): the policy to be used;
            actor_optimizer (dict): parameters to specify the actor
                optimizer algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            batch_size (int): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            warmup_transitions (int): number of samples to accumulate in the
                replay memory to start the policy fitting;
            tau (float): value of coefficient for soft updates;
            lr_alpha (float): Learning rate for the entropy coefficient;
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

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator,
                                                     **target_critic_params)

        self._init_target(self._critic_approximator,
                          self._target_critic_approximator)

        self._log_alpha = torch.tensor(0., dtype=torch.float32)

        if policy.use_cuda:
            self._log_alpha = self._log_alpha.cuda().requires_grad_()
        else:
            self._log_alpha.requires_grad_()

        self._alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)

        self._add_save_attr(
            _critic_fit_params='pickle',
            _batch_size='primitive',
            _warmup_transitions='primitive',
            _tau='primitive',
            _target_entropy='primitive',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _log_alpha='torch',
            _alpha_optim='torch'
        )

        super().__init__(mdp_info, policy, actor_optimizer, policy.parameters())

    def fit(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(self._batch_size)

            if self._replay_memory.size > self._warmup_transitions:
                action_new = self.policy.draw_action_t(state)
                log_prob = self.policy.log_prob_t(state, action_new)
                loss = self._loss(state, action_new, log_prob)
                self._optimize_actor_parameters(loss)
                self._update_alpha(log_prob.detach())

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            self._critic_approximator.fit(state, action, q,
                                          **self._critic_fit_params)

            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)

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
        a = self.policy.draw_action(next_state)
        log_prob_next = self.policy.log_prob(next_state, a)

        q = self._target_critic_approximator.predict(
            next_state, a, prediction='min') - self._alpha_np * log_prob_next
        q *= 1 - absorbing

        return q

    def _post_load(self):
        if self._optimizer is not None:
            self._parameters = list(self.policy.parameters())

    @property
    def _alpha(self):
        return self._log_alpha.exp()

    @property
    def _alpha_np(self):
        return self._alpha.detach().cpu().numpy()
