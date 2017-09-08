import numpy as np

from mushroom.approximators.regressor import Regressor


class ActionRegressor(object):
    """
    This class is used to approximate the Q-function with a different
    approximator of the provided class for each action. It is often used in MDPs
    with discrete actions and cannot be used in MDPs with continuous actions.

    """
    def __init__(self, approximator, discrete_actions,
                 input_preprocessor=None, output_preprocessor=None,
                 state_action_preprocessor=None, **params):
        """
        Constructor.

        Args:
            approximator (object): the model class to approximate the
                Q-function of each action;
            discrete_actions ([int, list, np.array]): the action values to
                consider to do regression. If an integer number n is provided,
                the values of the actions ranges from 0 to n - 1;
            state_action_preprocessor (list, None): list of preprocessing steps
                to apply to the input data of each regressor of the action
                regressor;
            state_action_preprocessor (list, None): list of preprocessing steps
                to apply to the input of the action regressor;
            **params (dict): parameters dictionary to create each regressor.

        """
        if isinstance(discrete_actions, int):
            self.discrete_actions = np.arange(discrete_actions).reshape(-1, 1)
        else:
            self.discrete_actions = np.array(discrete_actions)
            if self.discrete_actions.ndim == 1:
                self.discrete_actions = self.discrete_actions.reshape(-1, 1)
            assert self.discrete_actions.ndim == 2
        self.models = list()

        if state_action_preprocessor is not None:
            self._preprocessor = state_action_preprocessor
        else:
            self._preprocessor = []

        for i in xrange(self.discrete_actions.shape[0]):
            self.models.append(
                Regressor(approximator,
                          input_preprocessor=input_preprocessor,
                          output_preprocessor=output_preprocessor,
                          **params)
            )

    def fit(self, x, y, **fit_params):
        """
        Fit the model.

        Args:
            x (list): a two elements list with states and actions;
            y (np.array): targets;
            **fit_params (dict): other parameters.

        """
        x = self._preprocess(x)

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
        x = self._preprocess(x)

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
        x = self._preprocess(x)

        for i in xrange(len(self.models)):
            idxs = np.argwhere((x[1] == i)[:, 0]).ravel()
            if idxs.size:
                y_0 = self.models[i].predict(x[0][idxs, :])
                break
        y = np.zeros((x[0].shape[0],) + y_0.shape[1:])
        y[idxs] = y_0
        for i in xrange(i + 1, len(self.models)):
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
        n_actions = self.discrete_actions.shape[0]

        sa = [x, self.discrete_actions[0:1]]
        y_0 = self.models[0].predict(self._preprocess(sa)[0])
        y = np.zeros((n_states, n_actions) + y_0.shape[1:])
        y[:, 0] = y_0
        for action in xrange(1, n_actions):
            sa = [x, self.discrete_actions[action:action + 1]]
            y[:, action] = self.models[action].predict(self._preprocess(sa)[0])

        return y

    def _preprocess(self, x):
        for p in self._preprocessor:
            x = p(x)

        return x

    def __str__(self):
        return str(self.models[0]) + ' with action regression.'
