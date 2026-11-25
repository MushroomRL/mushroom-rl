import math

import torch
import torch.optim as optim

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import OffPolicyDeepAC
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.rl_utils.replay_memory import ReplayMemory
from mushroom_rl.utils.torch_utils import TorchUtils
from mushroom_rl.rl_utils.parameters import Parameter

from copy import deepcopy


class SAC(OffPolicyDeepAC):
    """
    Soft Actor-Critic algorithm.
    "Soft Actor-Critic Algorithms and Applications".
    Haarnoja T. et al. 2019.

    """

    def __init__(self, mdp_info, policy, actor_optimizer, critic_params, batch_size, initial_replay_size,
                 max_replay_size, warmup_transitions, tau, lr_alpha, use_log_alpha_loss=False, target_entropy=None,
                 critic_fit_params=None, history_length=1, device=None):
        """
        Constructor.

        Args:
            policy (TorchPolicy): torch policy to be learned by the algorithm;
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
            target_entropy (float, None): target entropy for the policy, if None a default value is computed;
            critic_fit_params (dict, None): parameters of the fitting algorithm of the critic approximator;
            history_length (int, 1): number of consecutive observations stacked as policy input;
            device (str, None): device the agent data is stored on, or ``None`` for the default one.

        """
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = Parameter.make(batch_size, backend='torch', device=device)
        self._warmup_transitions = Parameter.make(warmup_transitions, backend='torch', device=device)
        self._tau = Parameter.make(tau, backend='torch', device=device)

        self._use_log_alpha_loss = use_log_alpha_loss

        if target_entropy is None:
            self._target_entropy = -math.prod(mdp_info.action_space.shape)
        else:
            self._target_entropy = target_entropy

        if 'n_models' in critic_params.keys():
            assert critic_params['n_models'] == 2
        else:
            critic_params['n_models'] = 2

        if device is not None:
            critic_params.setdefault('device', device)

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = TorchApproximator(**critic_params)
        self._target_critic_approximator = TorchApproximator(**target_critic_params)

        self._init_target(self._critic_approximator, self._target_critic_approximator)

        self._log_alpha = torch.tensor(0., dtype=torch.float32, requires_grad=True,
                                       device=TorchUtils.get_device(device))

        self._alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)

        super().__init__(mdp_info, policy, actor_optimizer, policy.parameters(), history_length=history_length,
                         device=device)

        self._replay_memory = ReplayMemory(mdp_info, self.info, initial_replay_size, max_replay_size,
                                           history_manager=self.history_manager)

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
        self._history_manager.update_preprocessors(dataset)
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            batch = self._replay_memory.get(self._batch_size())
            batch = self._augment_batch(batch)

            if self._replay_memory.size > self._warmup_transitions():
                action_new, log_prob = self.policy.draw_with_log_prob(batch.state)
                loss = self._loss(batch, action_new, log_prob)
                self._optimize_actor_parameters(loss)
                alpha_loss = self._update_alpha(log_prob.detach())

                if self._logger:
                    self._logger.log_training('actor', loss=loss.item(), entropy=-log_prob.mean().item())
                    self._logger.log_training('alpha', value=self._alpha.item(), loss=alpha_loss.item())

            with torch.no_grad():
                q = batch.reward + self.mdp_info.gamma * self._next_q(batch)

            self._critic_approximator.fit(batch.critic_state, batch.action, q, **self._critic_fit_params)

            self._update_target(self._critic_approximator, self._target_critic_approximator)

            if self._logger:
                self._logger.advance_step()

    def _loss(self, batch, action_new, log_prob):
        q_0 = self._critic_approximator(batch.critic_state, action_new, idx=0)
        q_1 = self._critic_approximator(batch.critic_state, action_new, idx=1)

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

    def _next_q(self, batch):
        """
        Target value of the next states of a batch.

        Args:
            batch (Batch): the fit batch, whose next states are evaluated.

        Returns:
            Action-values returned by the target critic for the next states and the actions returned by the actor,
            zero on the absorbing states.

        """
        a, log_prob_next = self.policy.draw_with_log_prob(batch.next_state)

        q = self._target_critic_approximator.predict(batch.critic_next_state, a, prediction='min') \
            - self._alpha * log_prob_next
        q *= ~batch.absorbing

        return q

    def _post_load(self, device):
        super()._post_load(device)
        self._update_alpha_optimizer_parameters()

    def _update_alpha_optimizer_parameters(self):
        if self._alpha_optim is not None:
            TorchUtils.update_optimizer_parameters(self._alpha_optim, [self._log_alpha])

    @property
    def _alpha(self):
        return self._log_alpha.exp()
