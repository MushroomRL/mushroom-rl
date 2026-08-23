import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.modules.activation as _activation_module
import torch.nn.modules.rnn as _rnn_module


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
    def get_optimizer(name, learning_rate, eps=None, decay=None):
        """
        Returns the dictionary describing a torch optimizer, as expected by the approximators, from a
        string name.

        Args:
            name (str): name of the optimizer (case-insensitive), among ``'adam'``, ``'adadelta'``,
                ``'rmsprop'`` and ``'rmspropcentered'``;
            learning_rate (float): the learning rate of the optimizer;
            eps (float, None): the epsilon term of the optimizer. If None, the torch default is used;
            decay (float, None): the smoothing constant of the rmsprop variants. If None, the torch
                default is used.

        Returns:
            The dictionary with the optimizer class and its parameters.

        Raises:
            ValueError: if the string does not match any of the supported optimizers.

        """
        params = dict(lr=learning_rate)

        if eps is not None:
            params['eps'] = eps

        name_lower = name.lower()

        if name_lower == 'adam':
            optimizer_class = optim.Adam
        elif name_lower == 'adadelta':
            optimizer_class = optim.Adadelta
        elif name_lower in ('rmsprop', 'rmspropcentered'):
            optimizer_class = optim.RMSprop

            if decay is not None:
                params['alpha'] = decay

            if name_lower == 'rmspropcentered':
                params['centered'] = True
        else:
            raise ValueError(f"Cannot find optimizer '{name}'. "
                             f"Available: ['adam', 'adadelta', 'rmsprop', 'rmspropcentered']")

        return {'class': optimizer_class, 'params': params}

    @staticmethod
    def compute_output_shape(module, input_shape):
        """
        Computes the output shape of a module for a given input shape, without running any real
        computation or consuming random state.

        Args:
            module (nn.Module): the module whose output shape has to be computed;
            input_shape (tuple): the shape of a single, unbatched input.

        Returns:
            The shape of a single, unbatched output, as a tuple.

        """
        meta_module = deepcopy(module).to('meta')
        dummy_input = torch.zeros(1, *input_shape, device='meta')
        output = meta_module(dummy_input)

        return tuple(output.shape[1:])

    @staticmethod
    def compute_flat_output_size(module, input_shape):
        """
        Computes the flattened output size of a module for a given input shape, without running any
        real computation or consuming random state.

        Args:
            module (nn.Module): the module whose output size has to be computed;
            input_shape (tuple): the shape of a single, unbatched input.

        Returns:
            The size of a single, unbatched, flattened output, as an integer.

        """
        return math.prod(TorchUtils.compute_output_shape(module, input_shape))

    @staticmethod
    def update_optimizer_parameters(optimizer, new_parameters):
        if len(optimizer.state) > 0:
            for p_old, p_new in zip(optimizer.param_groups[0]['params'], new_parameters):
                data = optimizer.state[p_old]
                del optimizer.state[p_old]
                optimizer.state[p_new] = data

        optimizer.param_groups[0]['params'] = new_parameters
