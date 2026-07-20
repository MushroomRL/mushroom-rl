import math

import torch
import torch.optim as optim

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.policy import SquashedGaussianTorchPolicy
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.rl_utils.replay_memory import ReplayMemory
from mushroom_rl.utils.torch_utils import TorchUtils
from mushroom_rl.rl_utils.parameters import Parameter

from copy import deepcopy
from itertools import chain


class SAC(DeepAC):
    """
    Soft Actor-Critic algorithm.
    "Soft Actor-Critic Algorithms and Applications".
    Haarnoja T. et al. 2019.

    """

    def __init__(self, mdp_info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha, use_log_alpha_loss=False,
                 log_std_min=-20, log_std_max=2, target_entropy=None, critic_fit_params=None):
        """
        Constructor.

        Args:
            actor_mu_params (dict): parameters of the actor mean approximator to build;
            actor_sigma_params (dict): parameters of the actor sigma approximator to build;
            actor_optimizer (dict): parameters to specify the actor optimizer algorithm;
            critic_params (dict): parameters of the critic approximator to build;
            batch_size ((int, Parameter)): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before starting the learning;
            max_replay_size (int): the maximum number of samples in the replay memory;
            warmup_transitions ([int, Parameter]): number of samples to accumulate in the replay memory to start the
                policy fitting;
            tau ([float, Parameter]): value of coefficient for soft updates;
            lr_alpha ([float, Parameter]): Learning rate for the entropy coefficient;
            use_log_alpha_loss (bool, False): whether to use the original implementation loss or the one from the
                paper;
            log_std_min ([float, Parameter]): Min value for the policy log std;
            log_std_max ([float, Parameter]): Max value for the policy log std;
            target_entropy (float, None): target entropy for the policy, if None a default value is computed;
            critic_fit_params (dict, None): parameters of the fitting algorithm of the critic approximator.

        """
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = Parameter.make(batch_size)
        self._warmup_transitions = Parameter.make(warmup_transitions)
        self._tau = Parameter.make(tau)

        self._use_log_alpha_loss = use_log_alpha_loss

        if target_entropy is None:
            self._target_entropy = -math.prod(mdp_info.action_space.shape)
        else:
            self._target_entropy = target_entropy

        if 'n_models' in critic_params.keys():
            assert critic_params['n_models'] == 2
        else:
            critic_params['n_models'] = 2

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = TorchApproximator(**critic_params)
        self._target_critic_approximator = TorchApproximator(**target_critic_params)

        actor_mu_approximator = TorchApproximator(**actor_mu_params)
        actor_sigma_approximator = TorchApproximator(**actor_sigma_params)

        policy = SquashedGaussianTorchPolicy(actor_mu_approximator, actor_sigma_approximator,
                                             mdp_info.action_space.low, mdp_info.action_space.high,
                                             log_std_min, log_std_max)

        self._init_target(self._critic_approximator, self._target_critic_approximator)

        self._log_alpha = torch.tensor(0., dtype=torch.float32, requires_grad=True)

        self._alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)

        policy_parameters = chain(actor_mu_approximator.parameters(), actor_sigma_approximator.parameters())

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

        self._replay_memory = ReplayMemory(mdp_info, self.info, initial_replay_size, max_replay_size)

        self._add_save_attr(
            _critic_fit_params='pickle',
            _batch_size='mushroom',
            _warmup_transitions='mushroom',
            _tau='mushroom',
            _target_entropy='primitive',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _use_log_alpha_loss='primitive',
            _log_alpha='torch',
            _alpha_optim='torch'
        )
        self._add_logger_attr('_critic_approximator', group='critic')

    def fit(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, *_ = self._replay_memory.get(self._batch_size())

            if self._replay_memory.size > self._warmup_transitions():
                action_new, log_prob = self.policy.draw_with_log_prob(state)
                loss = self._loss(state, action_new, log_prob)
                self._optimize_actor_parameters(loss)
                alpha_loss = self._update_alpha(log_prob.detach())

                if self._logger:
                    self._logger.log_training('actor', loss=loss.item(), entropy=-log_prob.mean().item())
                    self._logger.log_training('alpha', value=self._alpha.item(), loss=alpha_loss.item())

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            self._critic_approximator.fit(state, action, q.detach(), **self._critic_fit_params)

            self._update_target(self._critic_approximator, self._target_critic_approximator)

            if self._logger:
                self._logger.advance_step()

    def _loss(self, state, action_new, log_prob):
        q_0 = self._critic_approximator(state, action_new, idx=0)
        q_1 = self._critic_approximator(state, action_new, idx=1)

        q = torch.min(q_0, q_1)

        return (self._alpha * log_prob - q).mean()

    def _update_alpha(self, log_prob):
        if self._use_log_alpha_loss:
            alpha_loss = - (self._log_alpha * (log_prob + self._target_entropy)).mean()
        else:
            alpha_loss = - (self._alpha * (log_prob + self._target_entropy)).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

        return alpha_loss

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (torch.Tensor): the states where next action has to be evaluated;
            absorbing (torch.Tensor): the absorbing flag for the states in ``next_state``.

        Returns:
            Action-values returned by the critic for ``next_state`` and the action returned by the actor.

        """
        with torch.no_grad():
            a, log_prob_next = self.policy.draw_with_log_prob(next_state)

        q = self._target_critic_approximator.predict(next_state, a, prediction='min') - self._alpha * log_prob_next
        q *= ~absorbing

        return q

    def _post_load(self):
        self._update_optimizer_parameters(self.policy.parameters())
        self._update_alpha_optimizer_parameters()

    def _update_alpha_optimizer_parameters(self):
        if self._alpha_optim is not None:
            TorchUtils.update_optimizer_parameters(self._alpha_optim, [self._log_alpha])

    @property
    def _alpha(self):
        return self._log_alpha.exp()
