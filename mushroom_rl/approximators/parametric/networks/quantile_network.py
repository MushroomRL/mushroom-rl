import torch
import torch.nn as nn


class QuantileNetwork(nn.Module):
    """
    Distributional network for Quantile Regression DQN (QR-DQN), approximating the value distribution of each
    action with ``n_quantiles`` quantiles whose mean gives the Q-value.

    """
    def __init__(self, input_shape, output_shape, features_network, n_quantiles,
                 n_features, **kwargs):
        """
        Constructor.

        Args:
            input_shape (tuple): shape of the input (the state);
            output_shape (tuple): shape of the output (the number of actions);
            features_network (nn.Module): the network used to compute the features;
            n_quantiles (int): number of quantiles used to approximate the value distribution;
            n_features (int): number of features extracted by the features network;
            **kwargs: parameters forwarded to the features network.

        """
        super().__init__()

        self._n_output = output_shape[0]
        self._phi = features_network(input_shape, (n_features,),
                                     n_features=n_features, **kwargs)
        self._n_quantiles = n_quantiles

        self._quant = nn.ModuleList(
            [nn.Linear(n_features, n_quantiles) for _ in range(self._n_output)])

        for i in range(self._n_output):
            nn.init.xavier_uniform_(self._quant[i].weight,
                                    gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None, get_quantiles=False, **kwargs):
        features = self._phi(state)

        a_quant = [self._quant[i](features) for i in range(self._n_output)]
        a_quant = torch.stack(a_quant, dim=1)

        if not get_quantiles:
            quant = a_quant.mean(-1)

            if action is not None:
                return torch.squeeze(quant.gather(1, action))
            else:
                return quant
        else:
            if action is not None:
                action = torch.unsqueeze(
                    action.long(), 2).repeat(1, 1, self._n_quantiles)

                return torch.squeeze(a_quant.gather(1, action))
            else:
                return a_quant
