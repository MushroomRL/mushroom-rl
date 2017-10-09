import numpy as np
from sklearn.exceptions import NotFittedError

from mushroom.approximators.regressor import Regressor
from mushroom.approximators.action_regressor import ActionRegressor
from mushroom.utils.table import Table


class Ensemble(object):
    """
    This class implements functions to manage regressor ensembles.

    """
    def __init__(self, approximator, n_models, prediction='mean',
                 use_action_regressor=False, discrete_actions=None,
                 input_preprocessor=None, output_preprocessor=None,
                 state_action_preprocessor=None, **params):
        """
        Constructor.

        Args:
            approximator (object): the model class to approximate the
                Q-function of each action;
            n_models (int): number of models in the ensemble;
            prediction (str, 'mean'): type of prediction to return;
            use_action_regressor (bool, False): whether each single regressor in
                the ensemble is an action regressor or not.
            discrete_actions ([int, list, np.array], None): the action values to
                consider to do regression. If an integer number n is provided,
                the values of the actions ranges from 0 to n - 1.
            **params (dict): parameters dictionary to construct each regressor.

        """
        self._prediction = prediction
        self._use_action_regressor = use_action_regressor
        self._models = list()

        if self._use_action_regressor:
            params['state_action_preprocessor'] = state_action_preprocessor
            regressor_class = ActionRegressor
        else:
            regressor_class = Regressor

        for _ in xrange(n_models):
            self._models.append(
                regressor_class(approximator, discrete_actions=discrete_actions,
                                input_preprocessor=input_preprocessor,
                                output_preprocessor=output_preprocessor,
                                **params))

    def predict(self, x, compute_variance=False, **predict_params):
        """
        Predict.

        Args:
            x (list): a two elements list with states and actions.
            compute_variance (bool, False): whether to return variance of the
                prediction.
            **predict_params (dict): other params.

        Returns:
            The predictions of the model.

        """
        predictions = list()
        for i in xrange(len(self._models)):
            try:
                predictions.append(self._models[i].predict(x, **predict_params))
            except NotFittedError:
                pass

        if len(predictions) == 0:
            raise NotFittedError

        if self._prediction == 'mean':
            results = np.mean(predictions, axis=0)
        elif self._prediction == 'sum':
            results = np.sum(predictions, axis=0)
        else:
            raise ValueError
        if compute_variance:
            results = [results] + np.var(predictions, ddof=1, axis=0)

        return results

    def predict_all(self, x, compute_variance=False, **predict_params):
        """
        Predict for each action given a state.

        Args:
            x (np.array): states.
            compute_variance (bool): whether to return variance of the
                prediction.
            **predict_params (dict): other params.

        Returns:
            The predictions of the model.

        """
        predictions = list()
        for i in xrange(len(self._models)):
            try:
                predictions.append(
                    self._models[i].predict_all(x, **predict_params))
            except NotFittedError:
                pass

        if len(predictions) == 0:
            raise NotFittedError

        if self._prediction == 'mean':
            results = np.mean(predictions, axis=0)
        elif self._prediction == 'sum':
            results = np.sum(predictions, axis=0)
        else:
            raise ValueError
        if compute_variance:
            results = [results] + np.var(predictions, ddof=1, axis=0)

        return results

    def __len__(self):
        return len(self._models)

    def __getitem__(self, idx):
        return self._models[idx]

    def __str__(self):
        s = '.' if self._use_action_regressor else ' with action regression.'
        return 'Ensemble of %d ' % len(self) + str(self._models[0]) + s


class EnsembleTable(Ensemble):
    """
    This class implements functions to manage table ensembles.

    """
    def __init__(self, n_models, shape, prediction='mean'):
        """
        Constructor.

        Args:
            n_models (int): number of models in the ensemble;
            shape (np.array): shape of each table in the ensemble;
            prediction (str, 'mean'): type of prediction to return.

        """
        self._prediction = prediction
        self._models = list()

        for _ in xrange(n_models):
            self._models.append(Table(shape))
