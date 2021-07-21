import numpy as np
import torch
from sklearn.exceptions import NotFittedError

from mushroom_rl.core import Serializable


class Ensemble(Serializable):
    """
    This class is used to create an ensemble of regressors.

    """
    def __init__(self, model, n_models, prediction='mean', **params):
        """
        Constructor.

        Args:
            approximator (class): the model class to approximate the
                Q-function.
            n_models (int): number of regressors in the ensemble;
            prediction (str, ['mean', 'sum', 'min', 'max']): the type of
                prediction to make;
            **params: parameters dictionary to create each regressor.

        """
        self._model = list()
        self._prediction = prediction

        for _ in range(n_models):
            self._model.append(model(**params))

        self._add_save_attr(
            _model=self._get_serialization_method(model),
            _prediction='primitive'
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

    def predict(self, *z, idx=None, prediction=None, compute_variance=False,
                **predict_params):
        """
        Predict.

        Args:
            *z: a list containing the inputs to use to predict with each
                regressor of the ensemble;
            idx (int, None): index of the model to use for prediction;
            prediction (str, None): the type of prediction to make. When
                provided, it overrides the ``prediction`` class attribute;
            compute_variance (bool, False): whether to compute the variance
                of the prediction or not;
            **predict_params: other parameters used by the predict method
                the regressor.

        Returns:
            The predictions of the model.

        """
        if idx is None:
            idx = [x for x in range(len(self))]

        if isinstance(idx, int):
            try:
                results = self[idx].predict(*z, **predict_params)
            except NotFittedError:
                raise NotFittedError
        else:
            predictions = list()
            for i in idx:
                try:
                    predictions.append(self[i].predict(*z, **predict_params))
                except NotFittedError:
                    raise NotFittedError

            prediction = self._prediction if prediction is None else prediction
            if isinstance(predictions[0], np.ndarray):
                predictions = np.array(predictions)
            else:
                predictions = torch.stack(predictions, axis=0)

            if prediction == 'mean':
                results = predictions.mean(0)
            elif prediction == 'sum':
                results = predictions.sum(0)
            elif prediction == 'min':
                results = predictions.min(0)
            elif prediction == 'max':
                results = predictions.max(0)
            else:
                raise ValueError
            if compute_variance:
                results = [results, predictions.var(0)]

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
