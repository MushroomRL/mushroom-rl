import numpy as np

from mushroom.approximators.regressor import Regressor
from mushroom.approximators.action_regressor import ActionRegressor
from mushroom.utils.table import Table

class Ensemble(object):
    """
    This class implements functions to manage regressor ensembles.

    """
    def __init__(self, approximator, n_models, use_action_regressor=False,
                 discrete_actions=None, input_preprocessor=None,
                 output_preprocessor=None, state_action_preprocessor=None,
                 **params):
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
        self._n_models = n_models
        self._use_action_regressor = use_action_regressor
        self.models = list()

        if self._use_action_regressor:
            params['state_action_preprocessor'] = state_action_preprocessor
            regressor_class = ActionRegressor
        else:
            regressor_class = Regressor

        for _ in xrange(self._n_models):
            self.models.append(
                regressor_class(approximator, discrete_actions=discrete_actions,
                                input_preprocessor=input_preprocessor,
                                output_preprocessor=output_preprocessor,
                                **params))

    def predict(self, x, compute_variance=False, **predict_params):
        """
        Predict.

        Args:
            x (list): a two elements list with states and actions.

        Returns:
            The predictions of the model.

        """
        y_0 = self.models[0].predict(x, **predict_params)
        y = np.zeros((self._n_models,) + y_0.shape)
        y[0] = y_0
        for i, m in enumerate(self.models[1:]):
            y[i + 1] = m.predict(x, **predict_params)

        if compute_variance:
            return np.mean(y, axis=0), np.var(y, ddof=1, axis=0)
        else:
            return np.mean(y, axis=0)

    def predict_all(self, x, compute_variance=False, **predict_params):
        """
        Predict for each action given a state.

        Args:
            x (np.array): states.

        Returns:
            The predictions of the model.

        """
        y_0 = self.models[0].predict_all(x, **predict_params)
        y = np.zeros((self._n_models,) + y_0.shape)
        y[0] = y_0
        for i, m in enumerate(self.models[1:]):
            y[i + 1] = m.predict_all(x, **predict_params)

        if compute_variance:
            return np.mean(y, axis=0), np.var(y, ddof=1, axis=0)
        else:
            return np.mean(y, axis=0)

    def __len__(self):
        return self._n_models

    def __getitem__(self, idx):
        return self.models[idx]

    def __str__(self):
        s = '.' if self._use_action_regressor else ' with action regression.'
        return 'Ensemble of %d ' % self.n_models + str(self.models[0]) + s


class EnsembleTable(object):
    """
        This class implements functions to manage table ensembles.

        """

    def __init__(self, n_tables, shape):
        """
        Constructor.

        Args:
            tables (Table): tables in the ensemble;
        """

        self.tables = []
        for i in xrange(n_tables):
            self.tables.append(Table(shape))

    def predict(self, x, compute_variance=False):
        """
        Predict.

        Args:
            x (list): a two elements list with states and actions.

        Returns:
            The predictions of the model.

        """
        y_0 = self.tables[0].predict(x)
        y = np.zeros((len(self.tables),) + y_0.shape)
        y[0] = y_0
        for i, m in enumerate(self.tables[1:]):
            y[i + 1] = m.predict(x)

        if compute_variance:
            return np.mean(y, axis=0), np.var(y, ddof=1, axis=0)
        else:
            return np.mean(y, axis=0)

    def predict_all(self, x, compute_variance=False):
        """
        Predict for each action given a state.

        Args:
            x (np.array): states.

        Returns:
            The predictions of the model.

        """
        y_0 = self.tables[0].predict_all(x)
        y = np.zeros((len(self.tables),) + y_0.shape)
        y[0] = y_0
        for i, m in enumerate(self.tables[1:]):
            y[i + 1] = m.predict_all(x)

        if compute_variance:
            return np.mean(y, axis=0), np.var(y, ddof=1, axis=0)
        else:
            return np.mean(y, axis=0)

    def __len__(self):
        return len(self.tables)

    def __getitem__(self, idx):
        return self.tables[idx]

    def __str__(self):
        return 'Ensemble of %d ' % len(self.tables) + 'Tables'
