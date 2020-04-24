import numpy as np
from sklearn.exceptions import NotFittedError

from mushroom_rl.core import Serializable


class Ensemble(Serializable):
    """
    This class is used to create an ensemble of regressors.

    """
    def __init__(self, model, n_models, **params):
        """
        Constructor.

        Args:
            approximator (class): the model class to approximate the
                Q-function.
            n_models (int): number of regressors in the ensemble;
            **params: parameters dictionary to create each regressor.

        """
        self._model = list()

        for _ in range(n_models):
            self._model.append(model(**params))

        self._add_save_attr(
            _model=self._get_serialization_method(model)
        )

    def fit(self, *z, idx=None, **fit_params):
        """
        Fit the ``idx``-th model of the ensemble if ``idx`` is provided, every
        model otherwise.

        Args:
            *z: a list containing the inputs to use to predict with each
                regressor of the ensemble;
            idx (int, None): index of the model to fit;
            **fit_params: other params.

        """
        if idx is None:
            for i in range(len(self)):
                self[i].fit(*z, **fit_params)
        else:
            self[idx].fit(*z, **fit_params)

    def predict(self, *z, idx=None, prediction='mean', compute_variance=False,
                **predict_params):
        """
        Predict.

        Args:
            *z: a list containing the inputs to use to predict with each
                regressor of the ensemble;
            idx (int, None): index of the model to use for prediction;
            prediction (str, 'mean'): the type of prediction to make. It can
                be a 'mean' of the ensembles, or a 'sum';
            compute_variance (bool, False): whether to compute the variance
                of the prediction or not;
            **predict_params: other parameters used by the predict method
                the regressor.

        Returns:
            The predictions of the model.

        """
        if idx is None:
            predictions = list()
            for i in range(len(self._model)):
                try:
                    predictions.append(self[i].predict(*z, **predict_params))
                except NotFittedError:
                    pass

            if len(predictions) == 0:
                raise NotFittedError

            if prediction == 'mean':
                results = np.mean(predictions, axis=0)
            elif prediction == 'sum':
                results = np.sum(predictions, axis=0)
            elif prediction == 'min':
                results = np.amin(predictions, axis=0)
            else:
                raise ValueError
            if compute_variance:
                results = [results, np.var(predictions, ddof=1, axis=0)]
        else:
            try:
                results = self[idx].predict(*z, **predict_params)
            except NotFittedError:
                raise NotFittedError

        return results

    def reset(self):
        """
        Reset the model parameters.

        """
        try:
            for m in self.model:
                m.reset()
        except AttributeError:
            raise NotImplementedError('Attempt to reset weights of a'
                                      ' non-parametric regressor.')

    @property
    def model(self):
        """
        Returns:
            The list of the models in the ensemble.

        """
        return self._model

    def __len__(self):
        return len(self._model)

    def __getitem__(self, idx):
        return self._model[idx]
