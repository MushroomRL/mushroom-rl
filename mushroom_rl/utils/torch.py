import torch
import numpy as np


class TorchUtils(object):
    _default_device = 'cpu'

    @classmethod
    def set_default_device(cls, device):
        cls._default_device = device

    @classmethod
    def get_device(cls, device=None):
        return cls._default_device if device is None else device

    @classmethod
    def set_weights(cls, parameters, weights, device=None):
        """
        Function used to set the value of a set of torch parameters given a
        vector of values.

        Args:
            parameters (list): list of parameters to be considered;
            weights (numpy.ndarray): array of the new values for
                the parameters;
            device (str, None): device to use to store the tensor.

        """
        idx = 0
        for p in parameters:
            shape = p.data.shape

            c = 1
            for s in shape:
                c *= s

            w = weights[idx:idx + c].reshape(shape)

            w_tensor = torch.as_tensor(w, device=cls.get_device(device)).type(p.data.dtype)

            p.data = w_tensor
            idx += c

        # assert idx == weights.size # TODO check if you can put another guard here

    @staticmethod
    def get_weights(parameters):
        """
        Function used to get the value of a set of torch parameters as
        a single vector of values.

        Args:
            parameters (list): list of parameters to be considered.

        Returns:
            A numpy vector consisting of all the values of the vectors.

        """
        weights = list()

        for p in parameters:
            w = p.data.detach()
            weights.append(w.flatten())

        weights = torch.concatenate(weights)

        return weights

    @staticmethod
    def zero_grad(parameters):
        """
        Function used to set to zero the value of the gradient of a set
        of torch parameters.

        Args:
            parameters (list): list of parameters to be considered.

        """

        for p in parameters:
            if p.grad is not None:
               p.grad.detach_()
               p.grad.zero_()

    @staticmethod
    def get_gradient(params):
        """
        Function used to get the value of the gradient of a set of
        torch parameters.

        Args:
            parameters (list): list of parameters to be considered.

        """
        views = []
        for p in params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    @classmethod
    def to_float_tensor(cls, x, device=None):
        """
        Function used to convert a numpy array to a float torch tensor.

        Args:
            x (np.ndarray): numpy array to be converted as torch tensor;
            device (str, None): device to use to store the tensor.

        Returns:
            A float tensor build from the values contained in the input array.

        """
        return torch.as_tensor(x, device=cls.get_device(device), dtype=torch.float)

    @classmethod
    def to_int_tensor(cls, x, device=None):
        """
        Function used to convert a numpy array to a float torch tensor.

        Args:
            x (np.ndarray): numpy array to be converted as torch tensor;
            device (str, None): device to use to store the tensor.

        Returns:
            A float tensor build from the values contained in the input array.

        """
        return torch.as_tensor(x, device=cls.get_device(device), dtype=torch.int)

    @staticmethod
    def update_optimizer_parameters(optimizer, new_parameters):
        if len(optimizer.state) > 0:
            for p_old, p_new in zip(optimizer.param_groups[0]['params'], new_parameters):
                data = optimizer.state[p_old]
                del optimizer.state[p_old]
                optimizer.state[p_new] = data

        optimizer.param_groups[0]['params'] = new_parameters


class CategoricalWrapper(torch.distributions.Categorical):
    """
    Wrapper for the Torch Categorical distribution.

    Needed to convert a vector of mushroom discrete action in an input with the proper shape of the original
    distribution implemented in torch

    """
    def __init__(self, logits):
        super().__init__(logits=logits)

    def log_prob(self, value):
        return super().log_prob(value.squeeze())
