import numpy as np

from mushroom_rl.core import Serializable
from mushroom_rl.features import Features


class CMAC(Serializable):
    """
    This class implements a Cerebellar Model Arithmetic Computer.


    """
    def __init__(self, tiles, weights=None, output_shape=(1,), **kwargs):
        """
        Constructor.

        Args:
            tiles (list): list of tilings to discretize the input space.
            weights (np.ndarray): array of weights to initialize the weights
                of the approximator;
            input_shape (np.ndarray, None): the shape of the input of the
                model;
            output_shape (np.ndarray, (1,)): the shape of the output of the
                model;
            **kwargs: other params of the approximator.

        """
        assert len(output_shape) == 1

        output_dim = output_shape[0]

        self._phi = Features(tilings=tiles)
        self._n = len(tiles)

        if weights is not None:
            assert weights.size == output_dim * self._phi.size
            self._w = weights.reshape((output_dim, -1))
        else:
            self._w = np.zeros((output_dim, self._phi.size))

        self._add_save_attr(_w='numpy', _phi='pickle')

    def fit(self, x, y, **fit_params):
        """
        Fit the model.

        Args:
            x (np.ndarray): input;
            y (np.ndarray): target;
            **fit_params: other parameters used by the fit method of the
                regressor.

        """
        y_hat = self.predict(x)
        delta_y = np.atleast_2d(y - y_hat).T
        phi = np.atleast_2d(self._phi(x))
        delta_w = np.empty_like(self._w)
        for i in range(self._w.shape[0]):
            sum_phi = np.sum(phi, axis=0)
            sum_phi[sum_phi == 0] = 1.
            delta_w[i] = delta_y[i] @ phi / self._n / sum_phi
        self._w += delta_w

    def predict(self, x, **predict_params):
        """
        Predict.

        Args:
            x (np.ndarray): input;
            **predict_params: other parameters used by the predict method
                the regressor.

        Returns:
            The predictions of the model.

        """
        prediction = np.ones((x.shape[0], self._w.shape[0]))
        indexes = self._phi.compute_indexes(x)

        if x.shape[0] == 1:
            indexes = list([indexes])

        for i, idx in enumerate(indexes):
            prediction[i] = np.sum(self._w[:, idx], axis=-1)

        return prediction

    @property
    def weights_size(self):
        """
        Returns:
            The size of the array of weights.

        """
        return self._w.size

    def get_weights(self):
        """
        Getter.

        Returns:
            The set of weights of the approximator.

        """
        return self._w.flatten()

    def set_weights(self, w):
        """
        Setter.

        Args:
            w (np.ndarray): the set of weights to set.

        """
        self._w = w.reshape(self._w.shape)

    def diff(self, state, action=None):
        """
        Compute the derivative of the output w.r.t. ``state``, and ``action``
        if provided.

        Args:
            state (np.ndarray): the state;
            action (np.ndarray, None): the action.

        Returns:
            The derivative of the output w.r.t. ``state``, and ``action``
            if provided.

        """
        phi = self._phi(state)
        if len(self._w.shape) == 1 or self._w.shape[0] == 1:
            return phi
        else:
            n_phi = self._w.shape[1]
            n_outs = self._w.shape[0]

            if action is None:
                shape = (n_phi * n_outs, n_outs)
                df = np.zeros(shape)
                start = 0
                for i in range(n_outs):
                    stop = start + n_phi
                    df[start:stop, i] = phi
                    start = stop
            else:
                shape = (n_phi * n_outs)
                df = np.zeros(shape)
                start = action[0] * n_phi
                stop = start + n_phi
                df[start:stop] = phi

            return df

