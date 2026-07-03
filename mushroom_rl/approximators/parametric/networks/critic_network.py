import torch
import torch.nn as nn

from mushroom_rl.utils.torch_utils import TorchUtils


class CriticNetwork(nn.Module):
    """
    Simple critic network taking ``state`` and ``action`` as two separate inputs.

    """
    def __init__(self, input_shape, output_shape, n_features, n_layers=2,
                 activation='relu', gain_scale=1.0, weights_init='xavier',
                 bias_init=None, **kwargs):
        """
        Constructor.

        Args:
            input_shape (list): the network has two inputs, so this must be
                ``[state_shape, action_shape]``;
            output_shape (tuple): shape of the output (the Q-value);
            n_features ([int, list]): size of the hidden layers, or a list of sizes for each of them;
            n_layers (int, 2): number of hidden layers, used only if ``n_features`` is an int;
            activation (str, 'relu'): activation function for the hidden layers;
            gain_scale (float, 1.0): scaling factor for the weights initialization gain;
            weights_init (str, 'xavier'): weights initialization method;
            bias_init (str, None): bias initialization method;
            **kwargs: other parameters (unused).

        """
        super().__init__()

        assert isinstance(input_shape, list) and len(input_shape) == 2, \
            'CriticNetwork requires input_shape=[state_shape, action_shape].'

        n_input = input_shape[0][-1] + input_shape[1][-1]
        n_output = output_shape[0]

        if n_features is None:
            hidden_sizes = []
        elif isinstance(n_features, int):
            hidden_sizes = [n_features] * n_layers
        else:
            hidden_sizes = list(n_features)

        act_class = TorchUtils.get_activation(activation)
        self._activation = act_class()

        try:
            hidden_gain = nn.init.calculate_gain(act_class.__name__.lower()) * gain_scale
        except ValueError:
            hidden_gain = nn.init.calculate_gain('relu') * gain_scale
        output_gain = nn.init.calculate_gain('linear') * gain_scale

        layer_sizes = [n_input] + hidden_sizes + [n_output]
        self._layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])

        for layer in self._layers[:-1]:
            TorchUtils.init_weights(layer, hidden_gain, weights_init, bias_init)
        TorchUtils.init_weights(self._layers[-1], output_gain, weights_init, bias_init)

    def forward(self, state, action):
        x = torch.cat((state.float(), action.float()), dim=1)
        for layer in self._layers[:-1]:
            x = self._activation(layer(x))
        q = self._layers[-1](x)
        return torch.squeeze(q)