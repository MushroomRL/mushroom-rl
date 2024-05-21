import torch.nn as nn


class LinearNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, use_bias=False, gain=None, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._f = nn.Linear(n_input, n_output, bias=use_bias)

        if gain is None:
            gain = nn.init.calculate_gain('linear')

        nn.init.xavier_uniform_(self._f.weight, gain=gain)

    def forward(self, state, **kwargs):
        return self._f(state)
