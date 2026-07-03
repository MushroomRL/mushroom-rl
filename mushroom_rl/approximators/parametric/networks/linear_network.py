import torch.nn as nn


class LinearNetwork(nn.Module):
    """
    Single fully connected layer mapping the ``state`` to the network output, with no activation function.

    """
    def __init__(self, input_shape, output_shape, use_bias=False, gain=None, **kwargs):
        """
        Constructor.

        Args:
            input_shape (tuple): shape of the input (the state);
            output_shape (tuple): shape of the output;
            use_bias (bool, False): whether to add a bias term to the linear layer;
            gain (float, None): gain used for the weights initialization; if None, the linear gain is used;
            **kwargs: other parameters (unused).

        """
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._f = nn.Linear(n_input, n_output, bias=use_bias)

        if gain is None:
            gain = nn.init.calculate_gain('linear')

        nn.init.xavier_uniform_(self._f.weight, gain=gain)

    def forward(self, state, **kwargs):
        return self._f(state)
