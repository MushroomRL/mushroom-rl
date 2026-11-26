import torch
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.policy.policy import Policy

from mushroom_rl.rl_utils.parameters import Parameter


class TDPolicy(Policy):
    """
    Base class for the policies built on a Q-function.

    """
    def __init__(self, backend='numpy', device=None):
        """
        Constructor.

        Args:
            backend (str, 'numpy'): name of the array backend used by the policy;
            device (str, None): device the policy is stored on, or ``None`` for the default one.

        """
        super().__init__(backend, device)

        self._approximator = None
        self._n_actions = None
        self._predict_params = dict()

        self._add_save_attr(_approximator='mushroom!',
                            _n_actions='primitive',
                            _predict_params='pickle')

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

    def draw_action_greedy(self, state):
        """
        Return the action maximizing the Q-function in ``state``, or in each row of a batch of states, ties broken
        uniformly at random.

        Args:
            state: a single state, or a batch of ``n`` states.

        Returns:
            The greedy action, of shape ``(1,)``, or the greedy actions, of shape ``(n, 1)``.

        """
        with torch.no_grad():
            q = self._approximator.predict(state, **self._predict_params)

        return self._sample_greedy(q)

    def _sample_greedy(self, q):
        """
        Returns:
            The action maximizing each row of ``q``, ties broken uniformly at random: shape ``(1,)`` for a single
            state, ``(n, 1)`` for a batch of ``n`` states.

        """
        greedy = q == self._backend.expand_dims(self._backend.max(q, dim=-1), -1)

        return self._backend.multinomial(greedy / self._backend.expand_dims(self._backend.sum(greedy, dim=-1), -1))


class EpsGreedy(TDPolicy):
    """
    Epsilon greedy policy.

    """
    def __init__(self, epsilon, backend='numpy', device=None):
        """
        Constructor.

        Args:
            epsilon ([float, Parameter]): the exploration coefficient. It indicates
                the probability of performing a random actions in the current
                step;
            backend (str, 'numpy'): name of the array backend used by the policy;
            device (str, None): device the policy is stored on, or ``None`` for the default one.

        """
        super().__init__(backend, device)

        self._epsilon = Parameter.make(epsilon, backend=backend, device=self._device)

        self._add_save_attr(_epsilon='mushroom')
        self._add_logger_attr(_epsilon='epsilon', group='policy')

    def __call__(self, *args):
        """
        Compute the probabilities of the actions in a state, or in each row of a batch of states.

        Args:
            *args: the state, of shape ``(n_state,)``, or the batch of ``n`` states, of shape ``(n, n_state)``;
                optionally followed by the action, of shape ``(1,)``, or the actions, of shape ``(n, 1)``.

        Returns:
            Without an action, the probabilities of all actions: shape ``(n_actions,)`` for a single state,
            ``(n, n_actions)`` for a batch. With an action, the probability of that action, of shape ``()``, for a
            single state, the probability of each row's action, of shape ``(n,)``, for a batch.

        """
        state = args[0]
        with torch.no_grad():
            q = self._approximator.predict(state, **self._predict_params)

        greedy = q == self._backend.expand_dims(self._backend.max(q, dim=-1), -1)
        n_greedy = self._backend.expand_dims(self._backend.sum(greedy, dim=-1), -1)
        epsilon = self._epsilon.get_value(state)
        probs = epsilon / self._n_actions + greedy * (1. - epsilon) / n_greedy

        if len(args) == 2:
            action = args[1]
            if probs.ndim == 1:
                return probs[action][0]
            else:
                return probs[self._backend.arange(0, len(probs), device=self._device), action[:, 0]]

        return probs

    def draw_action(self, state):
        """
        Sample an action in ``state``, or one action per row of a batch of states. The exploration coefficient is
        updated once per call, whatever the number of states.

        Args:
            state: a single state, or a batch of ``n`` states.

        Returns:
            The sampled action, of shape ``(1,)``, or the sampled actions, of shape ``(n, 1)``.

        """
        with torch.no_grad():
            q = self._approximator.predict(state, **self._predict_params)
        epsilon = self._epsilon(state)

        greedy = self._sample_greedy(q)
        explore = self._backend.rand(*greedy.shape, device=self._device) < epsilon
        random = self._backend.randint(0, self._n_actions, greedy.shape, device=self._device)

        return self._backend.where(explore, random, greedy)

    def set_epsilon(self, epsilon):
        """
        Setter.

        Args:
            epsilon ([float, Parameter]): the exploration coefficient. It indicates the
            probability of performing a random actions in the current step.

        """
        self._epsilon = Parameter.make(epsilon, backend=self.backend_name, device=self._device)

    def update(self, *args):
        """
        Update the value of the epsilon parameter at the provided point (e.g. in
        case of different values of epsilon for each visited state according to
        the number of visits).

        Args:
            *args: point at which the parameter is updated.

        """
        self._epsilon.update(*args)


class Boltzmann(TDPolicy):
    """
    Boltzmann softmax policy.

    """
    def __init__(self, beta, backend='numpy', device=None):
        """
        Constructor.

        Args:
            beta ([float, Parameter]): the inverse of the temperature distribution. As
            the temperature approaches infinity, the policy becomes more and
            more random. As the temperature approaches 0.0, the policy becomes
            more and more greedy;
            backend (str, 'numpy'): name of the array backend used by the policy;
            device (str, None): device the policy is stored on, or ``None`` for the default one.

        """
        super().__init__(backend, device)
        self._beta = Parameter.make(beta, backend=backend, device=self._device)

        self._add_save_attr(_beta='mushroom')
        self._add_logger_attr(_beta='beta', group='policy')

    def __call__(self, *args):
        """
        Compute the probabilities of the actions in a state, or in each row of a batch of states.

        Args:
            *args: the state, of shape ``(n_state,)``, or the batch of ``n`` states, of shape ``(n, n_state)``;
                optionally followed by the action, of shape ``(1,)``, or the actions, of shape ``(n, 1)``.

        Returns:
            Without an action, the probabilities of all actions: shape ``(n_actions,)`` for a single state,
            ``(n, n_actions)`` for a batch. With an action, the probability of that action, of shape ``()``, for a
            single state, the probability of each row's action, of shape ``(n,)``, for a batch.

        """
        state = args[0]
        with torch.no_grad():
            q = self._approximator.predict(state, **self._predict_params)

        q_beta = q * self._beta.get_value(state)
        q_beta -= self._backend.expand_dims(self._backend.max(q_beta, dim=-1), -1)
        qs = self._backend.exp(q_beta)
        probs = qs / self._backend.expand_dims(self._backend.sum(qs, dim=-1), -1)

        if len(args) == 2:
            action = args[1]
            if probs.ndim == 1:
                return probs[action][0]
            else:
                return probs[self._backend.arange(0, len(probs), device=self._device), action[:, 0]]

        return probs

    def draw_action(self, state):
        """
        Sample an action in ``state``, or one action per row of a batch of states. The inverse temperature is
        updated once per call, whatever the number of states.

        Args:
            state: a single state, or a batch of ``n`` states.

        Returns:
            The sampled action, of shape ``(1,)``, or the sampled actions, of shape ``(n, 1)``.

        """
        self._beta.update(state)

        return self._backend.multinomial(self(state))

    def set_beta(self, beta):
        """
        Setter.

        Args:
            beta ((float, Parameter)): the inverse of the temperature distribution.

        """
        self._beta = Parameter.make(beta, backend=self.backend_name, device=self._device)

    def update(self, *args):
        """
        Update the value of the beta parameter at the provided point (e.g. in
        case of different values of beta for each visited state according to
        the number of visits).

        Args:
            *args: point at which the parameter is updated.

        """
        self._beta.update(*args)


class Mellowmax(TDPolicy):
    """
    Mellowmax policy.
    "An Alternative Softmax Operator for Reinforcement Learning". Asadi K. and
    Littman M.L. 2017.

    """
    def __init__(self, omega, beta_min=0., beta_max=10., backend='numpy', device=None):
        """
        Constructor.

        The root giving beta is unique, but it is unbounded, while Brent's method needs a bracketing
        interval: when the root falls outside ``[beta_min, beta_max]``, beta is clipped to the end of the
        interval it lies beyond. With a positive omega the root is never negative, because mellowmax is at
        least the mean of the values, so the interval starts at zero. Give a negative ``beta_min`` to search
        for the root of a negative omega, whose operator approaches the minimum rather than the maximum.

        Args:
            omega (Parameter): the omega parameter of the policy from which beta
                of the Boltzmann policy is computed;
            beta_min (float, 0.): one end of the bracketing interval for
                minimization with Brent's method;
            beta_max (float, 10.): the other end of the bracketing interval for
                minimization with Brent's method;
            backend (str, 'numpy'): name of the array backend used by the policy;
            device (str, None): device the policy is stored on, or ``None`` for the default one.

        """
        super().__init__(backend, device)

        self._omega = Parameter.make(omega, backend=backend, device=self._device)
        self._beta_min = beta_min
        self._beta_max = beta_max

        self._add_save_attr(_omega='mushroom', _beta_min='primitive', _beta_max='primitive')
        self._add_logger_attr(_omega='omega', group='policy')

    def __call__(self, *args):
        """
        Compute the probabilities of the actions in a state, or in each row of a batch of states.

        Args:
            *args: the state, of shape ``(n_state,)``, or the batch of ``n`` states, of shape ``(n, n_state)``;
                optionally followed by the action, of shape ``(1,)``, or the actions, of shape ``(n, 1)``.

        Returns:
            Without an action, the probabilities of all actions: shape ``(n_actions,)`` for a single state,
            ``(n, n_actions)`` for a batch. With an action, the probability of that action, of shape ``()``, for a
            single state, the probability of each row's action, of shape ``(n,)``, for a batch.

        """
        state = args[0]
        with torch.no_grad():
            q = self._approximator.predict(state, **self._predict_params)
        q_numpy = ArrayBackend.convert(q, to='numpy')
        omega = self._omega.get_value(state)

        if q.ndim == 1:
            beta = self._find_beta(q_numpy, omega)
        else:
            beta = ArrayBackend.convert(np.array([self._find_beta(q_i, omega) for q_i in q_numpy], dtype=q_numpy.dtype),
                                        to=self.backend_name, device=self._device)
            beta = self._backend.expand_dims(beta, -1)

        q_beta = q * beta
        q_beta -= self._backend.expand_dims(self._backend.max(q_beta, dim=-1), -1)
        qs = self._backend.exp(q_beta)
        probs = qs / self._backend.expand_dims(self._backend.sum(qs, dim=-1), -1)

        if len(args) == 2:
            action = args[1]
            if probs.ndim == 1:
                return probs[action][0]
            else:
                return probs[self._backend.arange(0, len(probs), device=self._device), action[:, 0]]

        return probs

    def draw_action(self, state):
        """
        Sample an action in ``state``, or one action per row of a batch of states. The omega parameter is updated
        once per call, whatever the number of states.

        Args:
            state: a single state, or a batch of ``n`` states.

        Returns:
            The sampled action, of shape ``(1,)``, or the sampled actions, of shape ``(n, 1)``.

        """
        self._omega.update(state)

        return self._backend.multinomial(self(state))

    def _find_beta(self, q, omega):
        """
        Returns:
            The inverse temperature of the Boltzmann policy matching the mellowmax of the Q-values ``q`` of a
            single state.

        """
        mm = (logsumexp(q * omega) - np.log(q.size)) / omega

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
            return self._beta_max if f(self._beta_max) < 0. else self._beta_min
