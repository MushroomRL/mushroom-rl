import numpy as np

from mushroom.utils.parameters import Parameter


class GaussianPolicy:
    def __init__(self, mu, sigma):
        self.__name__ = 'GaussianPolicy'

        assert isinstance(sigma, Parameter)

        self._approximator = mu
        self._sigma = sigma

    def __call__(self, *args):
        if len(args) == 1:
            return self._sample_action(args[0])
        elif len(args) == 2:
            return self._compute_prob(args[0], args[1])

        raise ValueError('args must be state, or state and action')

    def diff(self, state, action):
        return self._compute_prob(state, action) * self.diff_log(state, action)

    def diff_log(self, state, action):
        mu, sigma = self._compute_gaussian(state, False)
        delta = action - mu
        g_mu = np.expand_dims(self._approximator.diff(state), axis=1)

        g = np.dot(g_mu, delta) / sigma**2

        return np.expand_dims(g, axis=1)

    def set_sigma(self, sigma):
        assert isinstance(sigma, Parameter)

        self._sigma = sigma

    def update(self, *idx):
        self._sigma.update(*idx)

    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    @property
    def weights_shape(self):
        return self._approximator.weights_shape

    def _compute_gaussian(self, state, update=True):
        if update:
            sigma = self._sigma(state)
        else:
            sigma = self._sigma.get_value(state)
        mu = np.reshape(self._approximator.predict(np.expand_dims(state,
                                                                  axis=0)), -1)

        return mu, sigma

    def _sample_action(self, state):
        mu, sigma = self._compute_gaussian(state)

        return np.random.normal(mu, sigma)

    def _compute_prob(self, state, action):
        mu, sigma = self._compute_gaussian(state, False)

        return np.exp(-0.5 * (action - mu)**2 / sigma**2) / np.sqrt(
            2 * np.pi) / sigma

    def __str__(self):
        return self.__name__
