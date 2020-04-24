import numpy as np
from mushroom_rl.core import Serializable


class QRegressor(Serializable):
    """
    This class is used to create a regressor that approximates the Q-function
    using a multi-dimensional output where each output corresponds to the
    Q-value of each action. This is used, for instance, by the ``ConvNet`` used
    in examples/atari_dqn.

    """
    def __init__(self, approximator, **params):
        """
        Constructor.

        Args:
            approximator (class): the model class to approximate the
                Q-function;
            **params: parameters dictionary to the regressor.

        """
        self.model = approximator(**params)

        self._add_save_attr(
            model=self._get_serialization_method(approximator)
        )

    def fit(self, state, action, q, **fit_params):
        """
        Fit the model.

        Args:
            state (np.ndarray): states;
            action (np.ndarray): actions;
            q (np.ndarray): target q-values;
            **fit_params: other parameters used by the fit method of the
                regressor.

        """
        self.model.fit(state, action, q, **fit_params)

    def predict(self, *z, **predict_params):
        """
        Predict.

        Args:
            *z: a list containing states or states and actions depending
                on whether the call requires to predict all q-values or only
                one q-value corresponding to the provided action;
            **predict_params: other parameters used by the predict method
                of each regressor.

        Returns:
            The predictions of the model.

        """
        assert len(z) == 1 or len(z) == 2

        state = z[0]
        q = self.model.predict(state, **predict_params)

        if len(z) == 2:
            action = z[1].ravel()
            if q.ndim == 1:
                return q[action]
            else:
                return q[np.arange(q.shape[0]), action]
        else:
            return q

    def reset(self):
        """
        Reset the model parameters.

        """
        try:
            self.model.reset()
        except AttributeError:
            raise NotImplementedError('Attempt to reset weights of a'
                                      ' non-parametric regressor.')

    @property
    def weights_size(self):
        return self.model.weights_size

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, w):
        self.model.set_weights(w)

    def diff(self, state, action=None):
        if action is None:
            return self.model.diff(state)
        else:
            return self.model.diff(state, action).squeeze()

    def __len__(self):
        return len(self.model)
