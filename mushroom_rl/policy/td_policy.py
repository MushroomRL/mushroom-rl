import torch
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.policy.policy import Policy

from mushroom_rl.rl_utils.parameters import Parameter, to_parameter


class TDPolicy(Policy):
    def __init__(self, policy_state_shape=None, backend='numpy'):
        """
        Constructor.

        Args:
            backend (str, 'numpy'): name of the array backend used by the policy.

        """
        super().__init__(policy_state_shape)

        self._approximator = None
        self._n_actions = None
        self._predict_params = dict()
        self._backend = ArrayBackend.get_array_backend(backend)

        self._add_save_attr(_approximator='mushroom!',
                            _n_actions='primitive',
                            _predict_params='pickle',
                            _backend='primitive')

    def set_q(self, approximator):
        """
        Args:
            approximator (object): the approximator to use.

        """
        self._approximator = approximator
        if hasattr(approximator, 'n_actions'):
            self._n_actions = approximator.n_actions
        else:
            self._n_actions = approximator.output_shape[0]

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
    def __init__(self, epsilon, policy_state_shape=None, backend='numpy'):
        """
        Constructor.

        Args:
            epsilon ([float, Parameter]): the exploration coefficient. It indicates
                the probability of performing a random actions in the current
                step;
            backend (str, 'numpy'): name of the array backend used by the policy.

        """
        super().__init__(policy_state_shape, backend)

        self._epsilon = to_parameter(epsilon)

        self._add_save_attr(_epsilon='mushroom')

    def __call__(self, *args):
        state = args[0]
        with torch.no_grad():
            q = self._approximator.predict(self._backend.expand_dims(state, 0), **self._predict_params).ravel()
        max_a = self._backend.nonzero(q == q.max()).ravel()

        p = self._epsilon.get_value(state) / self._n_actions

        if len(args) == 2:
            action = args[1]
            if action in max_a:
                return p + (1. - self._epsilon.get_value(state)) / len(max_a)
            else:
                return p
        else:
            probs = self._backend.ones(self._n_actions) * p
            probs[max_a] += (1. - self._epsilon.get_value(state)) / len(max_a)

            return probs

    def draw_action(self, state, policy_state=None):
        if not self._backend.rand() < self._epsilon(state):
            with torch.no_grad():
                q = self._approximator.predict(state, **self._predict_params)
            max_a = self._backend.nonzero(q == q.max()).ravel()

            if len(max_a) > 1:
                max_a = max_a[self._backend.randint(0, len(max_a), (1,))]

            return max_a, None

        return self._backend.randint(0, self._n_actions, (1,)), None

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
    def __init__(self, beta, policy_state_shape=None, backend='numpy'):
        """
        Constructor.

        Args:
            beta ([float, Parameter]): the inverse of the temperature distribution. As
            the temperature approaches infinity, the policy becomes more and
            more random. As the temperature approaches 0.0, the policy becomes
            more and more greedy;
            backend (str, 'numpy'): name of the array backend used by the policy.

        """
        super().__init__(policy_state_shape, backend)
        self._beta = to_parameter(beta)

        self._add_save_attr(_beta='mushroom')

    def __call__(self, *args):
        state = args[0]
        with torch.no_grad():
            q = self._approximator.predict(state, **self._predict_params)
        q_beta = q * self._beta(state)
        q_beta -= q_beta.max()
        qs = self._backend.exp(q_beta)

        if len(args) == 2:
            action = args[1]

            return qs[action] / qs.sum()
        else:
            return qs / qs.sum()

    def draw_action(self, state, policy_state=None):
        return self._backend.multinomial(self(state)), None

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
            with torch.no_grad():
                q = self._outer._approximator.predict(state, **self._outer._predict_params)
            q = ArrayBackend.convert(q, to='numpy')
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

    def __init__(self, omega, beta_min=-10., beta_max=10., policy_state_shape=None, backend='numpy'):
        """
        Constructor.

        Args:
            omega (Parameter): the omega parameter of the policy from which beta
                of the Boltzmann policy is computed;
            beta_min (float, -10.): one end of the bracketing interval for
                minimization with Brent's method;
            beta_max (float, 10.): the other end of the bracketing interval for
                minimization with Brent's method;
            backend (str, 'numpy'): name of the array backend used by the policy.

        """
        beta_mellow = self.MellowmaxParameter(self, omega, beta_min, beta_max)

        super().__init__(beta_mellow, policy_state_shape, backend)

    def set_beta(self, beta):
        raise RuntimeError('Cannot change the beta parameter of Mellowmax policy')

    def update(self, *idx):
        raise RuntimeError('Cannot update the beta parameter of Mellowmax policy')
