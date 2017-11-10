import numpy as np

from mushroom.utils.parameters import Parameter


class TDPolicy(object):
    def __init__(self):
        """
        Constructor.

        """
        self._approximator = None

    def __call__(self, *args):
        """
        Compute the probability of taking action in a certain state following
        the policy.

        Args:
            *args (list): list containing a state or a state and an action.

        Returns:
            The probability of all actions following the policy in the given
            state if the list contains only the state, else the probability
            of the given action in the given state following the policy.

        """
        raise NotImplementedError

    def draw_action(self, state):
        """
        Sample an action in `state` using the policy.

        Args:
            state (np.array): the state where the agent is.

        Returns:
            The action sampled from the policy.

        """
        raise NotImplementedError

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
    def __init__(self, epsilon):
        """
        Constructor.

        Args:
            epsilon (Parameter): the exploration coefficient. It indicates
                the probability of performing a random actions in the current
                step.

        """
        self.__name__ = 'EpsGreedy'

        super(EpsGreedy, self).__init__()

        assert isinstance(epsilon, Parameter)
        self._epsilon = epsilon

    def __call__(self, *args):
        state = args[0]
        q = self._approximator.predict(np.expand_dims(state, axis=0)).ravel()
        max_a = np.argwhere(q == np.max(q)).ravel()

        p = self._epsilon.get_value(state) / float(self._approximator.n_actions)

        if len(args) == 2:
            action = args[1]
            if action in max_a:
                return p + (1. - self._epsilon.get_value(state)) / len(max_a)
            else:
                return p
        else:
            probs = np.ones(self._approximator.n_actions) * p
            probs[max_a] += (1. - self._epsilon.get_value(state)) / len(max_a)

            return probs

    def draw_action(self, state):
        if not np.random.uniform() < self._epsilon(state):
            q = self._approximator.predict(np.expand_dims(state, axis=0))
            max_a = np.argwhere((q == np.max(q, axis=1)).ravel()).ravel()

            if len(max_a) > 1:
                max_a = np.array([np.random.choice(max_a)])

            return max_a

        return np.array([np.random.choice(self._approximator.n_actions)])

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


class Softmax(TDPolicy):
    """
    Softmax policy using a Boltzmann distribution.

    """
    def __init__(self, tau):
        """
        Constructor.

        Args:
            tau (float): the temperature of the distribution. As the temperature
            approaches infinity, the policy becomes more and more random. As the
            temperature approaches 0.0, the policy becomes more and more greedy.

        """
        self.__name__ = 'Softmax'

        super(Softmax, self).__init__()
        self._tau = tau

    def __call__(self, *args):
        state = args[0]
        qs = np.ones(self._approximator.n_actions)
        for a in xrange(self._approximator.n_actions):
            qs[a] = (np.e**(self._approximator.predict(state, a) / self._tau))

        if len(args) == 2:
            action = args[1]

            return qs[action] / qs
        else:
            p = np.ones(qs.size)
            for i in xrange(p.size):
                p[i] = qs[i] / qs

            return p

    def draw_action(self, state):
        return np.array([np.random.choice(self._approximator.n_actions,
                                          p=self(state))])
