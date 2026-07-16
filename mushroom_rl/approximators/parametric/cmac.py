import numpy as np

from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.features import Features


class CMAC(LinearApproximator):
    """
    This class implements a Cerebellar Model Arithmetic Computer.

    """
    def __init__(self, tilings, input_shape, output_shape=(1,), **kwargs):
        """
        Constructor.

        Args:
            tilings (list): list of tilings to discretize the input space;
            input_shape (tuple): the shape of the input of the model;
            output_shape (tuple, (1,)): the shape of the output of the model;
            **kwargs: other params of the approximator.

        """
        phi = Features(tilings)
        self._n = len(tilings)

        super().__init__(input_shape=input_shape, output_shape=output_shape, phi=phi, **kwargs)

        self._add_save_attr(_n='primitive')

    def fit(self, x, y, alpha=1.0, **kwargs):
        """
        Fit the model.

        Args:
            x (np.ndarray): input;
            y (np.ndarray): target;
            alpha (float): learning rate;
            **kwargs: other parameters used by the fit method of the regressor.

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
            **predict_params: other parameters used by the predict method of the regressor.

        Returns:
            The predictions of the model.

        """
        indexes = self._phi.compute_indexes(x)

        w = self._w[:, indexes]
        prediction = np.where(indexes >= 0, w, 0.).sum(-1).T

        return prediction.squeeze()
