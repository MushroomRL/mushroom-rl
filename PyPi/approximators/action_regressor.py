import numpy as np

from PyPi.approximators.regressor import Regressor


class ActionRegressor(object):
    """
    This class is used to approximate the Q-function with a different
    approximator of the provided class for each action. It is often used in MDPs
    with discrete actions and cannot be used in MDPs with continuous actions.
    """
    def __init__(self, approximator_class, discrete_actions, **params):
        """
        Constructor.

        # Arguments
            approximator_class (object): the model class to approximate the
            Q-function of each action;
            discrete_actions (np.array): the values of the discrete actions;
            **params (dict): parameters dictionary to co each regressor.
        """
        self._discrete_actions = discrete_actions
        self._action_dim = self._discrete_actions.shape[1]
        self.models = list()

        for i in range(self._discrete_actions.shape[0]):
            self.models.append(Regressor(approximator_class, **params))

    def fit(self, x, y, **fit_params):
        """
        Fit the model.

        # Arguments
            x (np.array): input dataset containing states and actions;
            y (np.array): target;
            fit_params (dict): other parameters.
        """
        action_idx = x.shape[1] - self._action_dim
        for i in range(len(self.models)):
            action = self._discrete_actions[i]
            idxs = np.argwhere(
                (x[:, action_idx:] == action)[:, 0]).ravel()

            if idxs.size:
                self.models[i].fit(x[idxs, :action_idx], y[idxs], **fit_params)

    def predict(self, x):
        """
        Predict.

        # Arguments
            x (np.array): input dataset containing states and actions.

        # Returns
            The predictions of the model.
        """
        predictions = np.zeros((x.shape[0]))
        action_idx = x.shape[1] - self._action_dim
        for i in range(len(self.models)):
            action = self._discrete_actions[i]
            idxs = np.argwhere(
                (x[:, action_idx:] == action)[:, 0]).ravel()

            if idxs.size:
                predictions[idxs] = self.models[i].predict(x[idxs, :action_idx])

        return predictions

    def __str__(self):
        return str(self.models[0]) + ' with action regression.'
