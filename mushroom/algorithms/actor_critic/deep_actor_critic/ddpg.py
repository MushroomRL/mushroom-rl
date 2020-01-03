import numpy as np

from mushroom.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom.policy import Policy
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import TorchApproximator
from mushroom.utils.replay_memory import ReplayMemory

from copy import deepcopy


class DDPG(DeepAC):
    """
    Deep Deterministic Policy Gradient algorithm.
    "Continuous Control with Deep Reinforcement Learning".
    Lillicrap T. P. et al.. 2016.

    """
    def __init__(self, mdp_info, policy_class, policy_params,
                 actor_params, actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, tau, policy_delay=1,
                 critic_fit_params=None):
        """
        Constructor.

        Args:
            policy_class (Policy): class of the policy;
            policy_params (dict): parameters of the policy to build;
            actor_params (dict): parameters of the actor approximator to
                build;
            actor_optimizer (dict): parameters to specify the actor optimizer
                algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            batch_size (int): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            tau (float): value of coefficient for soft updates;
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

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

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
