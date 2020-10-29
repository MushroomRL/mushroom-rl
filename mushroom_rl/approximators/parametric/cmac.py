import numpy as np

from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.features import Features


class CMAC(LinearApproximator):
    """
    This class implements a Cerebellar Model Arithmetic Computer.


    """
    def __init__(self, tilings, weights=None, output_shape=(1,), **kwargs):
        """
        Constructor.

        Args:
            tilings (list): list of tilings to discretize the input space.
            weights (np.ndarray): array of weights to initialize the weights
                of the approximator;
            input_shape (np.ndarray, None): the shape of the input of the
                model;
            output_shape (np.ndarray, (1,)): the shape of the output of the
                model;
            **kwargs: other params of the approximator.

        """
        self._phi = Features(tilings=tilings)
        self._n = len(tilings)

        super().__init__(weights=weights, input_shape=(self._phi.size,), output_shape=output_shape)

        self._add_save_attr(_phi='pickle', _n='primitive')

    def fit(self, x, y, alpha=1.0, **kwargs):
        """
        Fit the model.

        Args:
            x (np.ndarray): input;
            y (np.ndarray): target;
            alpha (float): learning rate;
            **kwargs: other parameters used by the fit method of the
                regressor.

        """
        y_hat = self.predict(x)
        delta_y = np.atleast_2d(y - y_hat)
        if self._w.shape[0] > 1:
            delta_y = delta_y.T

        phi = np.atleast_2d(self._phi(x))
        sum_phi = np.sum(phi, axis=0)
        n = np.sum(phi, axis=1, keepdims=True)
        phi_n = phi / n
        sum_phi[sum_phi == 0] = 1.

        delta_w = delta_y @ phi_n / sum_phi
        self._w += alpha*delta_w

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

        return prediction.squeeze()

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
        return super().diff(phi, action)
