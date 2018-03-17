import numpy as np


class DeterministicPolicy:
    def __init__(self, mu):
        self.__name__ = 'DeterministicPolicy'
        self._approximator = mu

    def __call__(self, state, action):
        policy_action = self._approximator.predict(state)

        return 1. if np.array_equal(action, policy_action) else 0.

    def draw_action(self, state):
        return self._approximator.predict(state)

    def diff(self, state, action):
        raise RuntimeError('Deterministic policy is not differentiable')

    def diff_log(self, state, action):
        raise RuntimeError('Deterministic policy is not differentiable')

    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    @property
    def weights_size(self):
        return self._approximator.weights_size

    def __str__(self):
        return self.__name__
