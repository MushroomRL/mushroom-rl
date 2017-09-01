import numpy as np


class Regressor(object):
    """
    Regressor class used to preprocess input and output before passing them
    to the desired approximator.
    """
    def __init__(self, approximator, discrete_actions=None, preprocessor=None,
                 **params):
        """
        Constructor.

        # Arguments
            approximator (object): the approximator class to use;
            fit_action (bool, True): whether the model consider the action in
                the input sample or not;
            preprocessor (list, None): list of preprocessing step to apply to
                the input data
            params (dict): other parameters.
        """
        self.model = approximator(**params)

        if discrete_actions is not None:
            if isinstance(discrete_actions, int):
                self.discrete_actions = np.arange(
                    discrete_actions).reshape(-1, 1)
                self._actions_with_value = False
            else:
                self.discrete_actions = np.array(discrete_actions)
                if self.discrete_actions.ndim == 1:
                    self.discrete_actions =\
                        self.discrete_actions.reshape(-1, 1)
                assert self.discrete_actions.ndim == 2
                self._actions_with_value = True

        self._preprocessor = preprocessor if preprocessor is not None else []

    def fit(self, x, y, **fit_params):
        """
        Preprocess the input and output if requested and fit the model using
        its fit function.

        # Arguments
            x (np.array): input dataset containing states (and action, if
                action regression is not used);
            y (np.array): target;
            fit_params (dict): other parameters.
        """
        x, y = self._preprocess_fit(x, y)
        self.model.fit(x, y, **fit_params)

    def train_on_batch(self, x, y, **fit_params):
        """
        Preprocess the input and output if requested and fit the model on a
        single batch using its fit function.

        # Arguments
            x (np.array): input dataset containing states (and action, if
                action regression is not used);
            y (np.array): target;
            fit_params (dict): other parameters.
        """
        x, y = self._preprocess_fit(x, y)
        self.model.train_on_batch(x, y, **fit_params)

    def predict(self, x):
        """
        Preprocess the input and output if requested and make the prediction.

        # Arguments
            x (np.array): input dataset containing states (and action, if
                action regression is not used).

        # Returns
            The prediction of the model.
        """
        x = self._preprocess_predict(x)
        y = self.model.predict(x)

        return y

    def predict_all(self, x):
        """
        Predict Q-value for each action given a state.

        # Arguments
            x (np.array): input dataset containing states;
            actions (np.array): list of actions of the MDP.

        # Returns
            The predictions of the model.
        """
        if hasattr(self, 'discrete_actions'):
            assert x.ndim == 2

            n_states = x.shape[0]
            n_actions = self.discrete_actions.shape[0]
            action_dim = self.discrete_actions.shape[1]

            for p in self._preprocessor:
                x = p(x)

            y = np.zeros((n_states, n_actions))
            for action in xrange(n_actions):
                a = np.ones(
                    (n_states, action_dim)) * self.discrete_actions[action]
                samples = np.concatenate((x, a), axis=1)

                y[:, action] = self.model.predict(samples).ravel()
        else:
            for p in self._preprocessor:
                x = p(x)

            y = self.model.predict(x)

        return y

    def _preprocess_fit(self, x, y):
        return self._preprocess(x), y

    def _preprocess_predict(self, x):
        return self._preprocess(x)

    def _preprocess(self, x):
        if hasattr(self, 'discrete_actions'):
            assert isinstance(x, list) and len(x) == 2
            assert x[0].ndim == 2 and x[1].ndim == 2
            assert x[0].shape[0] == x[1].shape[0]

            if self._actions_with_value:
                x = np.concatenate((x[0], self.discrete_actions[x[1].ravel()]),
                                   axis=1)
            else:
                x = np.concatenate((x[0], x[1]), axis=1)

        for p in self._preprocessor:
            x = p(x)

        return x

    @property
    def shape(self):
        return self.model.shape

    def __str__(self):
        return str(self.model)
