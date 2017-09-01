import numpy as np

from mushroom.approximators.regressor import Regressor


class ActionRegressor(object):
    """
    This class is used to approximate the Q-function with a different
    approximator of the provided class for each action. It is often used in MDPs
    with discrete actions and cannot be used in MDPs with continuous actions.

    """
    def __init__(self, approximator, discrete_actions, **params):
        """
        Constructor.

        Args:
            approximator (object): the model class to approximate the
                Q-function of each action;
            discrete_actions ([int, list, np.array]): the action values to
                consider to do regression. If an integer number n is provided,
                the values of the actions ranges from 0 to n - 1.
            **params (dict): parameters dictionary to create each regressor.

        """
        if isinstance(discrete_actions, int):
            self._discrete_actions = np.arange(discrete_actions).reshape(-1, 1)
        else:
            self._discrete_actions = np.array(discrete_actions)
            if self._discrete_actions.ndim == 1:
                self._discrete_actions = self._discrete_actions.reshape(-1, 1)
            assert self._discrete_actions.ndim == 2
        self.models = list()

        for i in xrange(self._discrete_actions.shape[0]):
            self.models.append(Regressor(approximator, **params))

    def fit(self, x, y, **fit_params):
        """
        Fit the model.

        Args:
            x (list): a two elements list with states and actions;
            y (np.array): targets;
            **fit_params (dict): other parameters.

        """
        for i in xrange(len(self.models)):
            idxs = np.argwhere((x[1] == i)[:, 0]).ravel()

            if idxs.size:
                self.models[i].fit(x[0][idxs, :], y[idxs], **fit_params)

    def train_on_batch(self, x, y, **fit_params):
        """
        Fit the model on a single batch.

        Args:
            x (list): a two elements list with states and actions;
            y (np.array): targets;
            **fit_params (dict): other parameters.

        """
        for i in xrange(len(self.models)):
            idxs = np.argwhere((x[1] == i)[:, 0]).ravel()

            if idxs.size:
                self.models[i].train_on_batch(x[0][idxs, :],
                                              y[idxs],
                                              **fit_params)

    def predict(self, x):
        """
        Predict.

        Args:
            x (list): a two elements list with states and actions;

        Returns:
            The predictions of the model.

        """
        y = np.zeros((x[0].shape[0]))
        for i in xrange(len(self.models)):
            idxs = np.argwhere((x[1] == i)[:, 0]).ravel()

            if idxs.size:
                y[idxs] = self.models[i].predict(x[0][idxs, :])

        return y

    def predict_all(self, x):
        """
        Predict for each action given a state.

        Args:
            x (np.array): states;

        Returns:
            The predictions of the model.

        """
        n_states = x.shape[0]
        n_actions = self._discrete_actions.shape[0]
        y = np.zeros((n_states, n_actions))

        for action in xrange(n_actions):
            y[:, action] = self.models[action].predict(x)

        return y

    def __str__(self):
        return str(self.models[0]) + ' with action regression.'
