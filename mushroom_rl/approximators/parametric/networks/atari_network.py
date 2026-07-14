import torch
import torch.nn as nn
import torch.nn.functional as F

from mushroom_rl.utils.torch_utils import TorchUtils


class AtariNetwork(nn.Module):
    """
    Convolutional network for Atari from pixel observations, outputting the Q-values for every action.
    Builds an ``AtariFeatureNetwork`` internally as its feature extractor.

    """
    def __init__(self, input_shape, output_shape, n_features=512, **kwargs):
        """
        Constructor.

        Args:
            input_shape (tuple): shape of the input image (channels, height, width);
            output_shape (tuple): shape of the output (one Q-value per action);
            n_features (int, 512): size of the feature layer feeding the output layer;
            **kwargs: other parameters, forwarded to the internal ``AtariFeatureNetwork``.

        """
        super().__init__()

        n_output = output_shape[0]

        self._phi = AtariFeatureNetwork(input_shape, (n_features,), **kwargs)
        self._h5 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h5.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None, **kwargs):
        h = self._phi(state, **kwargs)
        q = self._h5(h)

        if action is None:
            return q
        else:
            return torch.squeeze(q.gather(1, action.long()))


class AtariFeatureNetwork(nn.Module):
    """
    Convolutional feature extractor for Atari, sharing the same body as ``AtariNetwork`` but returning the
    features instead of the Q-values. Used as ``features_network`` by the distributional networks, and built
    internally by ``AtariNetwork``.

    """
    def __init__(self, input_shape, output_shape, **kwargs):
        """
        Constructor.

        Args:
            input_shape (tuple): shape of the input image (channels, height, width);
            output_shape (tuple): shape of the output (the feature size);
            **kwargs: other parameters (unused).

        """
        super().__init__()

        n_input = input_shape[0]
        n_output = output_shape[0]

        self._h1 = nn.Conv2d(n_input, 32, kernel_size=8, stride=4)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_out_size = TorchUtils.compute_flat_output_size(nn.Sequential(self._h1, self._h2, self._h3),
                                                            input_shape)
        self._h4 = nn.Linear(conv_out_size, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, state, action=None, **kwargs):
        h = F.relu(self._h1(state.float() / 255.))
        h = F.relu(self._h2(h))
        h = F.relu(self._h3(h))
        return F.relu(self._h4(h.view(h.shape[0], -1)))
