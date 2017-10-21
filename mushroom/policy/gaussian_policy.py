import numpy as np

from mushroom.utils.parameters import Parameter


class GaussianPolicy:
    def __init__(self, sigma):
        self.__name__ = 'GaussianPolicy'

        assert isinstance(sigma, Parameter)

        self._sigma = sigma

    def __call__(self, approximator, *args):
        if len(args) == 1:
            return self._sample_action(approximator, args[0])
        elif len(args) == 2:
            return self._compute_prob(approximator, args[0]. args[1])

        raise ValueError('args must be state, or state and action')

    def diff(self, state, action, approximator):
        return self(approximator, state, action) * self.diff_log(state, action)

    def diff_log(self, state, action, approximator):
        sigma = self._sigma.get_value(state)
        mu = approximator.predict(np.expand_dims(state, axis=0))
        delta = action - mu
        gMu = approximator.diff(state)
        return 0.5 * gMu.dot(delta)/sigma**2

    def set_sigma(self, sigma):
        assert isinstance(sigma, Parameter)

        self._sigma = sigma

    def update(self, *idx):
        self._sigma.update(*idx)

    def __str__(self):
        return self.__name__

    def _sample_action(self, approximator, state):
        sigma = self._sigma(state)
        mu = approximator.predict(np.expand_dims(state, axis=0))
        return np.random.normal(mu, sigma)

    def _compute_prob(self, approximator, state, action):
        sigma = self._sigma.get_value(state)
        mu = approximator.predict(np.expand_dims(state, axis=0))
        return np.exp(-0.5 * (action - mu)**2/sigma**2)/np.sqrt(2*np.pi)/sigma

