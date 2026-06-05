import torch
import torch.nn as nn


class QuantileNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, features_network, n_quantiles,
                 n_features, **kwargs):
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

    def forward(self, state, action=None, get_quantiles=False):
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