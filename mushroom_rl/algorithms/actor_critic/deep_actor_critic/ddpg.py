from copy import deepcopy

import torch

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import OffPolicyDeepAC
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.rl_utils.replay_memory import ReplayMemory
from mushroom_rl.rl_utils.parameters import Parameter


class DDPG(OffPolicyDeepAC):
    """
    Deep Deterministic Policy Gradient algorithm.
    "Continuous Control with Deep Reinforcement Learning".
    Lillicrap T. P. et al. 2016.

    """

    def __init__(self, mdp_info, policy, actor_optimizer, critic_params, batch_size, initial_replay_size,
                 max_replay_size, tau, policy_delay=1, critic_fit_params=None, critic_predict_params=None,
                 history_length=1, device=None):
        """
        Constructor.

        Args:
            policy (HasLocation): policy to be learned by the algorithm;
            actor_optimizer (dict): parameters to specify the actor optimizer
                algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            batch_size ([int, Parameter]): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            tau ((float, Parameter)): value of coefficient for soft updates;
            policy_delay ([int, Parameter], 1): the number of updates of the critic after
                which an actor update is implemented;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator;
            critic_predict_params (dict, None): parameters for the prediction with the
                critic approximator;
            history_length (int, 1): number of consecutive observations stacked as policy input;
            device (str, None): device the agent data is stored on, or ``None`` for the default one.

        """
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params
        self._critic_predict_params = dict() if critic_predict_params is None else critic_predict_params

        if device is not None:
            critic_params.setdefault('device', device)

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = TorchApproximator(**critic_params)
        self._target_critic_approximator = TorchApproximator(**target_critic_params)

        self._init_target(self._critic_approximator, self._target_critic_approximator)
        self._target_policy = policy.copy()

        super().__init__(mdp_info, policy, actor_optimizer, policy.parameters(), history_length=history_length,
                         device=device)

        self._batch_size = Parameter.make(batch_size, backend='torch', device=device)
        self._tau = Parameter.make(tau, backend='torch', device=device)
        self._policy_delay = Parameter.make(policy_delay, backend='torch', device=device)
        self._fit_count = 0

        self._replay_memory = ReplayMemory(mdp_info, self.info, initial_replay_size, max_replay_size,
                                           history_manager=self.history_manager)

        self._add_save_attr(
            _critic_fit_params='pickle',
            _critic_predict_params='pickle',
            _batch_size='mushroom',
            _tau='mushroom',
            _policy_delay='mushroom',
            _fit_count='primitive',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _target_policy='mushroom'
        )
        self._add_logger_attr('_critic_approximator', group='critic')

    def fit(self, dataset):
        self._history_manager.update_preprocessors(dataset)
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            batch = self._replay_memory.get(self._batch_size())
            batch = self._augment_batch(batch)

            with torch.no_grad():
                q = batch.reward + self.mdp_info.gamma * self._next_q(batch)

            self._critic_approximator.fit(batch.critic_state, batch.action, q, **self._critic_fit_params)

            if self._fit_count % self._policy_delay() == 0:
                loss = self._loss(batch)
                self._optimize_actor_parameters(loss)

                if self._logger:
                    self._logger.log_training('actor', loss=loss.item())

            self._update_target(self._critic_approximator, self._target_critic_approximator)
            self._update_target(self.policy, self._target_policy)

            self._fit_count += 1

            if self._logger:
                self._logger.advance_step()

    def _loss(self, batch):
        action = self.policy.location(batch.state)
        q = self._critic_approximator(batch.critic_state, action, **self._critic_predict_params)

        return -q.mean()

    def _next_q(self, batch):
        """
        Target value of the next states of a batch.

        Args:
            batch (Batch): the fit batch, whose next states are evaluated.

        Returns:
            Action-values returned by the target critic for the next states and the actions returned by the actor,
            zero on the absorbing states.

        """
        a = self._target_policy.location(batch.next_state)

        q = self._target_critic_approximator.predict(batch.critic_next_state, a, **self._critic_predict_params)
        q *= ~batch.absorbing

        return q
