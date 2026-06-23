import torch
import torch.nn as nn
import torch.nn.modules.activation as _activation_module
import torch.nn.modules.rnn as _rnn_module
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
            A torch tensor consisting of all the parameter values concatenated
            into a single vector.

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
    def get_activation(activation):
        """
        Returns a PyTorch activation class from a string name or passes through
        a class directly.

        Args:
            activation (str or type): either a string name (case-insensitive,
                e.g. ``'relu'``, ``'tanh'``) or a ``nn.Module`` subclass.

        Returns:
            A ``nn.Module`` subclass corresponding to the requested activation.

        Raises:
            ValueError: if the string does not match any activation in
                ``torch.nn.modules.activation``.

        """
        if isinstance(activation, str):
            activations_lc = [a.lower() for a in _activation_module.__all__]
            act_lower = activation.lower()
            if act_lower not in activations_lc:
                raise ValueError(f"Cannot find activation '{activation}'. "
                                 f"Available: {_activation_module.__all__}")
            idx = activations_lc.index(act_lower)
            return getattr(_activation_module, _activation_module.__all__[idx])
        else:
            return activation

    @staticmethod
    def init_weights(layer, gain, weights_init='xavier', bias_init=None):
        """
        Initializes the weights and biases of a linear layer.

        Args:
            layer (nn.Linear): the layer to initialize;
            gain (float): the gain to use for weight initialization;
            weights_init (str): ``'xavier'`` for Xavier uniform or
                ``'orthogonal'`` for orthogonal initialization;
            bias_init (str, None): ``None`` to leave bias unchanged,
                ``'zeros'`` to initialize bias to zero.

        """
        if weights_init == 'xavier':
            nn.init.xavier_uniform_(layer.weight, gain=gain)
        elif weights_init == 'orthogonal':
            nn.init.orthogonal_(layer.weight, gain=gain)
        else:
            raise ValueError(f"Unknown weights_init '{weights_init}'. "
                             f"Use 'xavier' or 'orthogonal'.")

        if bias_init == 'zeros':
            nn.init.constant_(layer.bias, 0)
        elif bias_init is not None:
            raise ValueError(f"Unknown bias_init '{bias_init}'. "
                             f"Use None or 'zeros'.")

    @staticmethod
    def get_recurrent_network(rnn_type):
        """
        Returns a PyTorch recurrent network class from a string name.

        Args:
            rnn_type (str): name of the RNN type (case-insensitive,
                e.g. ``'rnn'``, ``'gru'``, ``'lstm'``).

        Returns:
            A ``nn.Module`` subclass corresponding to the requested RNN type.

        Raises:
            ValueError: if the string does not match any RNN in
                ``torch.nn.modules.rnn``.

        """
        rnn_lc = [r.lower() for r in _rnn_module.__all__]
        rnn_lower = rnn_type.lower()
        if rnn_lower not in rnn_lc:
            raise ValueError(f"Cannot find RNN type '{rnn_type}'. "
                             f"Available: {_rnn_module.__all__}")
        idx = rnn_lc.index(rnn_lower)
        return getattr(_rnn_module, _rnn_module.__all__[idx])

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
