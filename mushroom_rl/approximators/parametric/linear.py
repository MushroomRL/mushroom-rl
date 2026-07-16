import numpy as np

from mushroom_rl.approximators.approximator import Approximator


class LinearApproximator(Approximator):
    """
    This class implements a linear approximator.

    """
    def __init__(self, input_shape, output_shape=(1,), phi=None, **kwargs):
        """
        Constructor.

        Args:
             input_shape (tuple): the shape of the input of the model. If ``phi`` is given, the input is
                featurized internally, so the number of weights is given by the size of ``phi`` and not by
                this shape;
             output_shape (tuple, (1,)): the shape of the output of the model;
             phi (Features, None): features to extract from the state;
             **kwargs: other params of the approximator.

        """
        super().__init__(input_shape=input_shape, output_shape=output_shape)

        assert isinstance(input_shape, tuple) and isinstance(output_shape, tuple)
        assert len(input_shape) == 1 and len(output_shape) == 1

        feature_dim = phi.size if phi is not None else input_shape[0]
        output_dim = output_shape[0]

        self._w = np.zeros((output_dim, feature_dim))

        self._phi = phi

        self._add_save_attr(
            _w='numpy',
            _phi='mushroom'
        )

    def fit(self, x, y, **fit_params):
        phi = np.atleast_2d(self.phi(x))
        self._w = np.atleast_2d(np.linalg.pinv(phi).dot(y).T)

    def predict(self, x, **predict_params):
        phi = np.atleast_2d(self.phi(x))
        return np.atleast_1d((phi @ self._w.T).squeeze())

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

    def phi(self, x):
        if self._phi is not None:
            return self._phi(x)
        else:
            return x

    def diff(self, state, action=None):
        """
        Compute the derivative of the output w.r.t. ``state``, and ``action`` if provided.

        Args:
            state (np.ndarray): the state;
            action (np.ndarray, None): the action.

        Returns:
            The derivative of the output w.r.t. ``state``, and ``action``
            if provided.

        """
        if len(self._w.shape) == 1 or self._w.shape[0] == 1:
            return self.phi(state)
        else:
            n_phi = self._w.shape[1]
            n_outs = self._w.shape[0]

            if action is None:
                shape = (n_phi * n_outs, n_outs)
                df = np.zeros(shape)
                start = 0
                for i in range(n_outs):
                    stop = start + n_phi
                    df[start:stop, i] = self.phi(state)
                    start = stop
            else:
                shape = (n_phi * n_outs)
                df = np.zeros(shape)
                start = action[0] * n_phi
                stop = start + n_phi
                df[start:stop] = self.phi(state)

            return df

    @property
    def weights_size(self):
        """
        Returns:
            The size of the array of weights.

        """
        return self._w.size
