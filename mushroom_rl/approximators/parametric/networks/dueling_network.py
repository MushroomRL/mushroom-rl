import torch
import torch.nn as nn


class DuelingNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, features_network, n_features,
                 avg_advantage, **kwargs):
        super().__init__()

        self._avg_advantage = avg_advantage

        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,),
                                     n_features=n_features, **kwargs)

        self._A = nn.Linear(n_features, self._n_output)
        self._V = nn.Linear(n_features, 1)

        nn.init.xavier_uniform_(self._A.weight,
                                gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self._V.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None):
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
            q_acted = torch.squeeze(q.gather(1, action.long()))

            return q_acted