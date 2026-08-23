import math

import torch
import torch.nn as nn

from mushroom_rl.utils.torch_utils import TorchUtils


class FeedForwardNetwork(nn.Module):
    """
    Simple feedforward network mapping a single ``state`` input to the network output (e.g. an action for a
    policy head, or a scalar for a state-value critic). Any non-flat input (e.g. a stacked observation
    history) is flattened into a single feature vector before being fed to the first layer.

    """
    def __init__(self, input_shape, output_shape, n_features, n_layers=2,
                 activation='relu', gain_scale=1.0, weights_init='xavier',
                 bias_init=None, action_history_shape=None, **kwargs):
        """
        Constructor.

        Args:
            input_shape (tuple): shape of the input (the state), flattened if not already 1D;
            output_shape (tuple): shape of the output (e.g. the action);
            n_features ([int, list]): size of the hidden layers, or a list of sizes for each of them;
            n_layers (int, 2): number of hidden layers, used only if ``n_features`` is an int;
            activation (str, 'relu'): activation function for the hidden layers;
            gain_scale (float, 1.0): scaling factor for the weights initialization gain;
            weights_init (str, 'xavier'): weights initialization method;
            bias_init (str, None): bias initialization method;
            action_history_shape (tuple, None): shape of the previous-action input; when set, the flattened
                previous action is concatenated to the observation;
            **kwargs: other parameters (unused).

        """
        super().__init__()

        self._input_ndim = len(input_shape)
        self._action_history_shape = action_history_shape

        n_input = math.prod(input_shape)
        if action_history_shape is not None:
            n_input += math.prod(action_history_shape)
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

    def forward(self, state, action_history=None, **kwargs):
        x = state.float()
        x = x.reshape(-1) if x.dim() == self._input_ndim else x.reshape(x.shape[0], -1)
        if self._action_history_shape is not None:
            flat_action_history = action_history.float().flatten(
                start_dim=action_history.dim() - len(self._action_history_shape))
            x = torch.cat([x, flat_action_history], dim=-1)
        for layer in self._layers[:-1]:
            x = self._activation(layer(x))
        return self._layers[-1](x)


class ActorNetwork(FeedForwardNetwork):
    """
    Alias of :class:`FeedForwardNetwork`, kept for symmetry with
    :class:`~mushroom_rl.approximators.parametric.networks.critic_network.CriticNetwork` when the network is
    used as a policy head rather than a state-value critic.

    """
