import numpy as np
import torch

from mushroom_rl.approximators.approximator import Approximator
from mushroom_rl.approximators.parametric.torch_approximator import TorchApproximator
from mushroom_rl.utils.torch_utils import TorchUtils


class NumpyTorchApproximator(Approximator):
    """
    Numpy interface to a :class:`TorchApproximator`, for use with numpy backend algorithms. When ``n_models > 1``
    construction returns an ``Ensemble`` of ``NumpyTorchApproximator``.

    """
    def __init__(self, network, input_shape, output_shape, **params):
        """
        Constructor.

        Args:
            network (torch.nn.Module): the network class to use;
            input_shape (tuple): the shape of the input of the network;
            output_shape (tuple): the shape of the output of the network;
            **params: other parameters used to build the :class:`TorchApproximator`.

        """
        super().__init__(input_shape=input_shape, output_shape=output_shape, backend='numpy')

        self._model = TorchApproximator(network, input_shape, output_shape, **params)

        self._add_save_attr(_model='mushroom')

    def predict(self, *args, **kwargs):
        """
        Predict.

        Args:
            *args: the numpy inputs of the network;
            **kwargs: other parameters used by the predict method of the :class:`TorchApproximator`.

        Returns:
            The numpy predictions of the model.

        """
        torch_args = [torch.as_tensor(x, device=TorchUtils.get_device()) for x in args]
        return self._model.predict(*torch_args, **kwargs).detach().cpu().numpy()

    def fit(self, *args, n_epochs=None, weights=None, epsilon=None, patience=1, validation_split=1., **kwargs):
        """
        Fit the model.

        Args:
            *args: the numpy inputs and targets, as in :meth:`TorchApproximator.fit`;
            n_epochs (int, None): the number of training epochs;
            weights (torch.Tensor, None): the weights of each sample in the computation of the loss;
            epsilon (float, None): the coefficient used for early stopping;
            patience (float, 1.): the number of epochs to wait until stop the learning if not improving;
            validation_split (float, 1.): the percentage of the dataset to use as training set;
            **kwargs: other parameters used by the fit method of the :class:`TorchApproximator`.

        """
        torch_args = [torch.as_tensor(x, device=TorchUtils.get_device()) for x in args]
        self._model.fit(*torch_args, n_epochs=n_epochs, weights=weights, epsilon=epsilon, patience=patience,
                        validation_split=validation_split, **kwargs)

    def diff(self, *args, **kwargs):
        """
        Compute the derivative of the output w.r.t. the model parameters.

        Args:
            *args: the numpy inputs of the network;
            **kwargs: other parameters used by the diff method of the :class:`TorchApproximator`.

        Returns:
            The numpy derivative of the output w.r.t. the model parameters.

        """
        torch_args = [torch.as_tensor(np.atleast_2d(x), device=TorchUtils.get_device()) for x in args]
        gradient = self._model.diff(*torch_args, **kwargs)
        return gradient.detach().cpu().numpy()

    def set_logger(self, logger, prefix=None, label=None):
        """
        Attach the logger to the approximator, so that the loss of the model is logged during its ``fit``.

        Args:
            logger (Logger): the logger object;
            prefix (str, None): optional group prepended to the logged metric names;
            label (str, None): optional name used for the loss. Defaults to ``'loss'``.

        """
        super().set_logger(logger, prefix, label)
        self._model.set_logger(logger, prefix, label)

    def set_weights(self, weights):
        """
        Setter.

        Args:
            weights (np.ndarray): the set of weights to set.

        """
        self._model.set_weights(torch.as_tensor(weights))

    def get_weights(self):
        """
        Returns:
            The set of weights of the model, as a numpy array.

        """
        return self._model.get_weights().detach().cpu().numpy()

    @property
    def weights_size(self):
        """
        Returns:
            The size of the array of weights.

        """
        return self._model.weights_size

    @property
    def model(self):
        """
        Returns:
            The underlying :class:`TorchApproximator`.

        """
        return self._model
