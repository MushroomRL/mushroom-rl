import numpy as np

from mushroom.approximators.action_regressor import ActionRegressor
from mushroom.approximators.regressor import Regressor


class Ensemble(object):
    """
    This class implements functions to manage regressor ensembles.

    """
    def __init__(self, approximator, n_models, use_action_regressor=False,
                 discrete_actions=None, **params):
        """
        Constructor.

        Args:
            approximator (object): the model class to approximate the
                Q-function of each action;
            n_models (int): number of models in the ensemble;
            use_action_regressor (bool, False): whether each single regressor in
            the ensemble is an action regressor or not.
            discrete_actions ([int, list, np.array], None): the action values to
                consider to do regression. If an integer number n is provided,
                the values of the actions ranges from 0 to n - 1.
            **params (dict): parameters dictionary to construct each regressor.

        """
        self.n_models = n_models
        self._use_action_regressor = use_action_regressor
        self.models = list()

        regressor_class =\
            ActionRegressor if self._use_action_regressor else Regressor

        for _ in xrange(self.n_models):
            self.models.append(
                regressor_class(approximator, discrete_actions=discrete_actions,
                                **params))

    def predict(self, x, **predict_params):
        """
        Predict.

        Args:
            x (list): a two elements list with states and actions.

        Returns:
            The predictions of the model.

        """
        y_0 = self.models[0].predict(x, **predict_params)
        y = np.zeros((self.n_models,) + y_0.shape)
        y[0] = y_0
        for i, m in enumerate(self.models[1:]):
            y[i + 1] = m.predict(x, **predict_params)
        y = np.mean(y, axis=0)

        return y

    def predict_all(self, x, **predict_params):
        """
        Predict for each action given a state.

        Args:
            x (np.array): states.

        Returns:
            The predictions of the model.

        """
        y_0 = self.models[0].predict_all(x, **predict_params)
        y = np.zeros((self.n_models,) + y_0.shape)
        y[0] = y_0
        for i, m in enumerate(self.models[1:]):
            y[i + 1] = m.predict_all(x, **predict_params)
        y = np.mean(y, axis=0)

        return y

    def __getitem__(self, idx):
        return self.models[idx]

    def __str__(self):
        s = '.' if self._use_action_regressor else ' with action regression.'
        return 'Ensemble of %d ' % self.n_models + str(self.models[0]) + s
