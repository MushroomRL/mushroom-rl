from mushroom_rl.approximators.parametric.networks.mushroom_network import MushroomNetwork
from mushroom_rl.approximators.parametric.networks.noisy_linear import NoisyLinear


class NoisyNetwork(MushroomNetwork):
    """
    Network for Noisy DQN, outputting the Q-values through a noisy linear layer whose learnable noise provides
    state-dependent exploration.

    """
    def __init__(self, input_shape, output_shape, features_network, n_features, sigma_coeff=.5, **kwargs):
        """
        Constructor.

        Args:
            input_shape (tuple): shape of the input (the state);
            output_shape (tuple): shape of the output (the number of actions);
            features_network (nn.Module): the network used to compute the features;
            n_features (int): number of features extracted by the features network;
            sigma_coeff (float, .5): scaling coefficient for the initial noise standard deviation, also forwarded to
                the features network;
            **kwargs: parameters forwarded to the features network.

        """
        super().__init__()

        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,), n_features=n_features, sigma_coeff=sigma_coeff,
                                     **kwargs)

        self._Q = NoisyLinear(n_features, self._n_output, sigma_coeff)

    def forward(self, state, action=None, **kwargs):
        features = self._phi(state)
        q = self._Q(features)
        if action is None:
            return q
        else:
            q_acted = q.gather(1, action.long()).squeeze(1)

            return q_acted
