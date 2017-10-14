import numpy as np


class ActionRegressor:
    """
    This class is used to approximate the Q-function with a different
    approximator of the provided class for each action. It is often used in MDPs
    with discrete actions and should not be used in MDPs with continuous
    actions.

    """
    def __init__(self, approximator, n_actions, approximator_params, **params):
        """
        Constructor.

        Args:
            approximator (object): the model class to approximate the
                Q-function of each action;
            n_actions (int): number of different actions of the problem. It
                determines the number of different regressors in the action
                regressor;
            approximator_params (dict): parameters dictionary to create each
                regressor;
            **params (dict): parameters dictionary used by the action
                regressor.

        """
        self.model = list()
        self._n_actions = n_actions

        for i in xrange(self._n_actions):
            self.model.append(approximator(**approximator_params))

        self._input_preprocessor = params.get('input_preprocessor', list())
        self._output_preprocessor = params.get('output_preprocessor', list())

    def fit(self, s, a, q, **fit_params):
        """
        Fit the model.

        Args:
            s (np.array): states;
            a (np.array): actions;
            q (np.array): target q-values;
            **fit_params (dict): other parameters used by the fit method
                of each regressor.

        """
        s, q = self._preprocess(s, q)

        for i in xrange(len(self.model)):
            idxs = np.argwhere((a == i)[:, 0]).ravel()

            if idxs.size:
                self.model[i].fit(s[idxs, :], q[idxs], **fit_params)

    def predict(self, *z, **predict_params):
        """
        Predict.

        Args:
            *z (list): a list containing states or states and actions depending
                on whether the call requires to predict all q-values or only
                one q-value corresponding to the provided action;
            **predict_params (dict): other parameters used by the predict method
                of each regressor.

        Returns:
            The predictions of the model.

        """
        s = z[0]
        s = self._preprocess(s)

        if len(z) == 2:
            a = z[1]
            q = np.zeros((s.shape[0], 1))
            for i in xrange(self._n_actions):
                idxs = np.argwhere((a == i)[:, 0]).ravel()
                if idxs.size:
                    q[idxs] = self.model[i].predict(s[idxs, :],
                                                    **predict_params)
        else:
            assert len(z) == 1

            q = np.zeros((s.shape[0], self._n_actions))
            for i in xrange(self._n_actions):
                q[:, i] = self.model[i].predict(s, **predict_params)

        return q

    def get_weights(self):
        w = list()
        for m in self.model:
            w.append(m.get_weights())

        return w

    def set_weights(self, w):
        for m in self.model:
            m.set_weights(w)

    def _preprocess(self, s, q=None):
        for p in self._input_preprocessor:
            s = p(s)

        if q is not None:
            for p in self._output_preprocessor:
                q = p(q)

            return s, q
        return s

    def __str__(self):
        return 'ActionRegressor of ' + str(self.model[0]) + ' with ' +\
            self._n_actions + ' actions.'
