import numpy as np
from sklearn import preprocessing


class Regressor(object):
    """
    Regressor class used to preprocess input and output before passing them
    to the desired approximator.
    """
    def __init__(self, approximator, fit_action=True, **params):
        """
        Constructor.

        # Arguments
            approximator_class (object): the approximator class to use;
            params (dict): other parameters.
        """
        self.features = params.pop('features', None)
        self.input_scaled = params.pop('input_scaled', False)
        self.output_scaled = params.pop('output_scaled', False)

        self.model = approximator(**params)
        self.fit_action = fit_action

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
        if self.fit_action:
            assert isinstance(x, list) and len(x) == 2
            assert x[0].ndim == 2 and x[1].ndim == 2
            assert x[0].shape[0] == x[1].shape[0]

            x = np.concatenate((x[0], x[1]), axis=1)

        if isinstance(x, list):
            if self.features:
                x[0] = self.features.fit_transform(x[0])

            if self.input_scaled:
                self.pre_x = preprocessing.StandardScaler()
                x[0] = self.pre_x.fit_transform(x[0])
        else:
            if self.features:
                x = self.features.fit_transform(x)

            if self.input_scaled:
                self.pre_x = preprocessing.StandardScaler()
                x = self.pre_x.fit_transform(x)

        if self.output_scaled:
            self.pre_y = preprocessing.StandardScaler()
            y = self.pre_y.fit_transform(y.reshape(-1, 1))

        self.model.fit(x, y, **fit_params)

    def predict(self, x):
        """
        Preprocess the input and output if requested and make the prediction.

        # Arguments
            x (np.array): input dataset containing states (and action, if
                action regression is not used).

        # Returns
            The prediction of the model.
        """
        if self.fit_action:
            assert isinstance(x, list) and len(x) == 2
            assert x[0].ndim == 2 and x[1].ndim == 2
            assert x[0].shape[0] == x[1].shape[0]

            x = np.concatenate((x[0], x[1]), axis=1)

        if isinstance(x, list):
            if self.features:
                x[0] = self.features.transform(x[0])

            if self.input_scaled:
                self.pre_x = preprocessing.StandardScaler()
                x[0] = self.pre_x.transform(x[0])
        else:
            if self.features:
                x = self.features.transform(x)

            if self.input_scaled:
                self.pre_x = preprocessing.StandardScaler()
                x = self.pre_x.transform(x)

        y = self.model.predict(x)

        return self.pre_y.inverse_transform(y) if self.output_scaled else y

    def predict_all(self, x, actions):
        if self.features:
            x = self.features.transform(x)

        if self.input_scaled:
            self.pre_x = preprocessing.StandardScaler()
            x = self.pre_x.transform(x)

        n_states = x.shape[0]
        n_actions = actions.shape[0]
        action_dim = actions.shape[1]
        y = np.zeros((n_states, n_actions))
        for action in xrange(n_actions):
            a = np.ones((n_states, action_dim)) * actions[action]
            if self.fit_action:
                assert x.ndim == 2

                samples = np.concatenate((x, a), axis=1)
            else:
                samples = [x, a]

            predictions = self.model.predict(samples)
            if predictions.ndim > 1:
                assert action == 0
                y = predictions
                break
            y[:, action] = self.model.predict(samples)

        return self.pre_y.inverse_transform(y) if self.output_scaled else y

    def __str__(self):
        return str(self.model)
