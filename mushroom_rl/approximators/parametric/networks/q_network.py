import torch
import torch.nn as nn

from mushroom_rl.utils.torch_utils import TorchUtils


class QNetwork(nn.Module):
    """
    Simple feedforward network outputting the Q-values for every discrete action given the ``state``, or the
    Q-value of a specific ``action`` when one is provided.

    """
    def __init__(self, input_shape, output_shape, n_features, n_layers=2,
                 activation='relu', gain_scale=1.0, weights_init='xavier',
                 bias_init=None, **kwargs):
        """
        Constructor.

        Args:
            input_shape (tuple): shape of the input (the state);
            output_shape (tuple): shape of the output (one Q-value per discrete action);
            n_features ([int, list]): size of the hidden layers, or a list of sizes for each of them;
            n_layers (int, 2): number of hidden layers, used only if ``n_features`` is an int;
            activation (str, 'relu'): activation function for the hidden layers;
            gain_scale (float, 1.0): scaling factor for the weights initialization gain;
            weights_init (str, 'xavier'): weights initialization method;
            bias_init (str, None): bias initialization method;
            **kwargs: other parameters (unused).

        """
        super().__init__()

        n_input = input_shape[-1]
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

    def forward(self, state, action=None, **kwargs):
        x = state.float()
        for layer in self._layers[:-1]:
            x = self._activation(layer(x))
        q = self._layers[-1](x)

        if action is None:
            return q
        else:
            return torch.squeeze(q.gather(1, action.long()))