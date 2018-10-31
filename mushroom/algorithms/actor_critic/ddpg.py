from copy import deepcopy

import numpy as np

import torch.nn as nn
from mushroom.algorithms import Agent
from mushroom.approximators import Regressor
from mushroom.utils.replay_memory import ReplayMemory


class ActorLoss(nn.Module):
    """
    Class used to implement the loss function of the actor.
    
    """
    def __init__(self, critic):
        super().__init__()

        self._critic = critic

    def forward(self, action, state):
        q = self._critic.model.network(state, action)

        return -q.mean()


class DDPG(Agent):
    """
    Deep Deterministic Policy Gradient algorithm.
    "Continuous Control with Deep Reinforcement Learning".
    Lillicrap T. P. et al.. 2016.

    """
    def __init__(self, actor_approximator, critic_approximator, policy_class,
                 mdp_info, batch_size, initial_replay_size, max_replay_size,
                 tau, actor_params, critic_params, policy_params,
                 actor_fit_params=None, critic_fit_params=None):
        """
        Constructor.

        Args:
            actor_approximator (object): the approximator to use for the actor;
            critic_approximator (object): the approximator to use for the
                critic;
            policy_class (Policy): class of the policy;
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
            policy_params (dict): parameters of the policy to build;
            actor_fit_params (dict, None): parameters of the fitting algorithm
                of the actor approximator;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator;

        """
        self._actor_fit_params = dict() if actor_fit_params is None else actor_fit_params
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = batch_size
        self._tau = tau

        self._replay_memory = ReplayMemory(initial_replay_size, max_replay_size)

        self._n_updates = 0

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(critic_approximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(critic_approximator,
                                                     **target_critic_params)

        if 'loss' not in actor_params:
            actor_params['loss'] = ActorLoss(self._critic_approximator)

        target_actor_params = deepcopy(actor_params)
        self._actor_approximator = Regressor(actor_approximator,
                                             **actor_params)
        self._target_actor_approximator = Regressor(actor_approximator,
                                                    **target_actor_params)

        self._target_actor_approximator.model.set_weights(
            self._actor_approximator.model.get_weights())
        self._target_critic_approximator.model.set_weights(
            self._critic_approximator.model.get_weights())

        policy = policy_class(self._actor_approximator, **policy_params)
        super().__init__(policy, mdp_info)

    def fit(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ =\
                self._replay_memory.get(self._batch_size)

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            self._critic_approximator.fit(state, action, q,
                                          **self._critic_fit_params)
            self._actor_approximator.fit(state, state,
                                         **self._actor_fit_params)

            self._n_updates += 1

            self._update_target()

    def _update_target(self):
        """
        Update the target networks.

        """
        critic_weights = self._tau * self._critic_approximator.model.get_weights()
        critic_weights += (1 - self._tau) * self._target_critic_approximator.get_weights()
        self._target_critic_approximator.set_weights(critic_weights)

        actor_weights = self._tau * self._actor_approximator.model.get_weights()
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
