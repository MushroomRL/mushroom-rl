import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from .policy import Policy

from mushroom_rl.utils.parameters import Parameter, to_parameter


class TDPolicy(Policy):
    def __init__(self):
        """
        Constructor.

        """
        self._approximator = None
        self._predict_params = dict()

        self._add_save_attr(_approximator='mushroom!',
                            _predict_params='pickle')

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


class EpsGreedy(TDPolicy):
    """
    Epsilon greedy policy.

    """
    def __init__(self, epsilon):
        """
        Constructor.

        Args:
            epsilon ([float, Parameter]): the exploration coefficient. It indicates
                the probability of performing a random actions in the current
                step.

        """
        super().__init__()

        self._epsilon = to_parameter(epsilon)

        self._add_save_attr(_epsilon='mushroom')

    def __call__(self, *args):
        state = args[0]
        q = self._approximator.predict(np.expand_dims(state, axis=0), **self._predict_params).ravel()
        max_a = np.argwhere(q == np.max(q)).ravel()

        p = self._epsilon.get_value(state) / self._approximator.n_actions

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
            q = self._approximator.predict(state, **self._predict_params)
            max_a = np.argwhere(q == np.max(q)).ravel()

            if len(max_a) > 1:
                max_a = np.array([np.random.choice(max_a)])

            return max_a

        return np.array([np.random.choice(self._approximator.n_actions)])

    def set_epsilon(self, epsilon):
        """
        Setter.

        Args:
            epsilon ([float, Parameter]): the exploration coefficient. It indicates the
            probability of performing a random actions in the current step.

        """
        self._epsilon = to_parameter(epsilon)

    def update(self, *idx):
        """
        Update the value of the epsilon parameter at the provided index (e.g. in
        case of different values of epsilon for each visited state according to
        the number of visits).

        Args:
            *idx (list): index of the parameter to be updated.

        """
        self._epsilon.update(*idx)


class Boltzmann(TDPolicy):
    """
    Boltzmann softmax policy.

    """
    def __init__(self, beta):
        """
        Constructor.

        Args:
            beta ([float, Parameter]): the inverse of the temperature distribution. As
            the temperature approaches infinity, the policy becomes more and
            more random. As the temperature approaches 0.0, the policy becomes
            more and more greedy.

        """
        super().__init__()
        self._beta = to_parameter(beta)

        self._add_save_attr(_beta='mushroom')

    def __call__(self, *args):
        state = args[0]
        q_beta = self._approximator.predict(state, **self._predict_params) * self._beta(state)
        q_beta -= q_beta.max()
        qs = np.exp(q_beta)

        if len(args) == 2:
            action = args[1]

            return qs[action] / np.sum(qs)
        else:
            return qs / np.sum(qs)

    def draw_action(self, state):
        return np.array([np.random.choice(self._approximator.n_actions,
                                          p=self(state))])

    def set_beta(self, beta):
        """
        Setter.

        Args:
            beta ((float, Parameter)): the inverse of the temperature distribution.

        """
        self._beta = to_parameter(beta)

    def update(self, *idx):
        """
        Update the value of the beta parameter at the provided index (e.g. in
        case of different values of beta for each visited state according to
        the number of visits).

        Args:
            *idx (list): index of the parameter to be updated.

        """
        self._beta.update(*idx)


class Mellowmax(Boltzmann):
    """
    Mellowmax policy.
    "An Alternative Softmax Operator for Reinforcement Learning". Asadi K. and
    Littman M.L.. 2017.

    """
    class MellowmaxParameter(Parameter):
        def __init__(self, outer, omega, beta_min, beta_max):
            self._omega = omega
            self._outer = outer
            self._beta_min = beta_min
            self._beta_max = beta_max

            self._add_save_attr(
                _omega='primitive',
                _outer='primitive',
                _beta_min='primitive',
                _beta_max='primitive',
            )

        def __call__(self, state):
            q = self._outer._approximator.predict(state, **self._outer._predict_params)
            mm = (logsumexp(q * self._omega(state)) - np.log(
                q.size)) / self._omega(state)

            def f(beta):
                v = q - mm
                beta_v = beta * v
                beta_v -= beta_v.max()

                return np.sum(np.exp(beta_v) * v)

            try:
                beta = brentq(f, a=self._beta_min, b=self._beta_max)
                assert not (np.isnan(beta) or np.isinf(beta))

                return beta
            except ValueError:
                return 0.

    def __init__(self, omega, beta_min=-10., beta_max=10.):
        """
        Constructor.

        Args:
            omega (Parameter): the omega parameter of the policy from which beta
                of the Boltzmann policy is computed;
            beta_min (float, -10.): one end of the bracketing interval for
                minimization with Brent's method;
            beta_max (float, 10.): the other end of the bracketing interval for
                minimization with Brent's method.

        """
        beta_mellow = self.MellowmaxParameter(self, omega, beta_min, beta_max)

        super().__init__(beta_mellow)

    def set_beta(self, beta):
        raise RuntimeError('Cannot change the beta parameter of Mellowmax policy')

    def update(self, *idx):
        raise RuntimeError('Cannot update the beta parameter of Mellowmax policy')
