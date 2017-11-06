import numpy as np

from mushroom.utils.parameters import Parameter


class TDPolicy(object):
    def __init__(self, observation_space, action_space):
        """
        Constructor.

        Args:
            observation_space (object): the state space;
            action_space (object): the action_space.

        """
        self.observation_space = observation_space
        self.action_space = action_space

        self._approximator = None

    def set_q(self, approximator):
        """
        Args:
            approximator (object): the approximator to use.

        """
        self._approximator = approximator

    def get_q(self):
        """
        Returns:
             The approximator used by the policy.

        """
        return self._approximator

    def __str__(self):
        return self.__name__


class EpsGreedy(TDPolicy):
    """
    Epsilon greedy policy.

    """
    def __init__(self, epsilon, observation_space, action_space):
        """
        Constructor.

        Args:
            observation_space (object): the state space;
            action_space (object): the action_space;
            epsilon (Parameter): the exploration coefficient. It indicates
                the probability of performing a random actions in the current
                step.

        """
        self.__name__ = 'EpsGreedy'

        super(EpsGreedy, self).__init__(observation_space, action_space)

        assert isinstance(epsilon, Parameter)
        self._epsilon = epsilon

    def __call__(self, state, action):
        """
        Compute the probability of taking `action` in `state` according to the
        policy.

        Args:
            state (np.array): the state where the agent is;
            action (np.array): the action whose probability has to be returned.

        Returns:
            The probability of taking `action`.

        """
        q = self._approximator.predict(np.expand_dims(state, axis=0))
        max_a = np.argwhere((q == np.max(q, axis=1)).ravel()).ravel()

        p = self._epsilon.get_value(state) / self.action_space.n
        if action in max_a:
            return p + (1. - self._epsilon.get_value(state)) / len(max_a)
        else:
            return p

    def draw_action(self, state):
        """
        Sample an action in `state` using the policy.

        Args:
            state (np.array): the state where the agent is.

        Returns:
            The action sampled from the policy.

        """
        if not np.random.uniform() < self._epsilon(state):
            q = self._approximator.predict(np.expand_dims(state, axis=0))
            max_a = np.argwhere((q == np.max(q, axis=1)).ravel()).ravel()

            if len(max_a) > 1:
                max_a = np.array([np.random.choice(max_a)])

            return max_a

        return self.action_space.sample()

    def set_epsilon(self, epsilon):
        """
        Setter.

        Args:
            epsilon (Parameter): the exploration coefficient. It indicates the
            probability of performing a random actions in the current step.

        """
        assert isinstance(epsilon, Parameter)

        self._epsilon = epsilon

    def update(self, *idx):
        """
        Update the value of the epsilon parameter (e.g. in case of different
        values of epsilon for each visited state according to the number of
        visits).

        Args:
            idx (int): value to use to update epsilon.

        """
        self._epsilon.update(*idx)
