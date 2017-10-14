from copy import deepcopy

import numpy as np
from sklearn.exceptions import NotFittedError

from mushroom.utils.table import Table


class Ensemble(object):
    """
    This class is used to create an ensemble of regressors.

    """
    def __init__(self, approximator, n_models, prediction='mean'):
        """
        Constructor.

        Args:
            approximator (object): the model class to approximate the
                Q-function.
            n_models (int): number of regressors in the ensemble;
            prediction (str): the type of prediction to make.

        """
        self._prediction = prediction
        self._model = list()

        for _ in xrange(n_models):
            self._model.append(deepcopy(approximator))

    def predict(self, *z, **predict_params):
        """
        Predict.

        Args:
            *z (list): a list containing the inputs to use to predict with each
                regressor of the ensemble;
            **predict_params (dict): other params.

        Returns:
            The predictions of the model.

        """
        predictions = list()
        for i in xrange(len(self._model)):
            try:
                predictions.append(self._model[i].predict(*z, **predict_params))
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
        if predict_params.get('compute_variance', False):
            results = [results] + np.var(predictions, ddof=1, axis=0)

        return results

    def __len__(self):
        return len(self._model)

    def __getitem__(self, idx):
        return self._model[idx]


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
        super(EnsembleTable, self).__init__(Table(shape), n_models, prediction)
