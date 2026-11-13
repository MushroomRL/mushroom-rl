import torch.nn as nn

from mushroom_rl.approximators.parametric.networks.mushroom_network import MushroomNetwork


class DuelingNetwork(MushroomNetwork):
    """
    Dueling architecture for DQN, splitting the shared features into a state-value stream and an advantage
    stream that are recombined into the Q-values.

    """
    def __init__(self, input_shape, output_shape, features_network, n_features, avg_advantage, **kwargs):
        """
        Constructor.

        Args:
            input_shape (tuple): shape of the input (the state);
            output_shape (tuple): shape of the output (the number of actions);
            features_network (nn.Module): the network used to compute the features;
            n_features (int): number of features extracted by the features network;
            avg_advantage (bool): whether to subtract the mean (True) or the max (False) advantage;
            **kwargs: parameters forwarded to the features network.

        """
        super().__init__()

        self._avg_advantage = avg_advantage

        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,),
                                     n_features=n_features, **kwargs)

        self._A = nn.Linear(n_features, self._n_output)
        self._V = nn.Linear(n_features, 1)

        self._init_weights()

    def forward(self, state, action=None, **kwargs):
        features = self._phi(state)

        advantage = self._A(features)
        value = self._V(features)

        q = value + advantage
        if self._avg_advantage:
            q -= advantage.mean(1).reshape(-1, 1)
        else:
            q -= advantage.max(1).values.reshape(-1, 1)

        if action is None:
            return q
        else:
            q_acted = q.gather(1, action.long()).squeeze(1)

            return q_acted

    def _init_weights(self):
        nn.init.xavier_uniform_(self._A.weight,
                                gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self._V.weight,
                                gain=nn.init.calculate_gain('linear'))
