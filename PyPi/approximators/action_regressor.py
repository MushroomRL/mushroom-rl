from copy import deepcopy

import numpy as np


class ActionRegressor(object):
    def __init__(self, model, discrete_actions):
        self._discrete_actions = discrete_actions
        self._action_dim = self._discrete_actions.shape[1]
        self.models = list()
        for i in range(self._discrete_actions.shape[0]):
            self.models.append(deepcopy(model))

    def fit(self, x, y):
        for i in range(len(self.models)):
            action = self._discrete_actions[i]
            idxs = np.argwhere(
                (x[:, -self._action_dim:] == action)[:, 0]).ravel()

            if idxs.size:
                self.models[i].fit(x[idxs, :-self._action_dim], y[idxs])

    def predict(self, x):
        predictions = np.zeros((x.shape[0]))
        for i in range(len(self.models)):
            action = self._discrete_actions[i]
            idxs = np.argwhere(
                (x[:, -self._action_dim:] == action)[:, 0]).ravel()

            if idxs.size:
                predictions[idxs] = self.models[i].predict(
                    x[idxs, :-self._action_dim])

        return predictions
