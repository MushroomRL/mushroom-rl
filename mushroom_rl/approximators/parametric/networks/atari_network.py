import torch
import torch.nn as nn
import torch.nn.functional as F


class AtariNetwork(nn.Module):
    n_features = 512

    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[0]
        n_output = output_shape[0]

        self._h1 = nn.Conv2d(n_input, 32, kernel_size=8, stride=4)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self._h4 = nn.Linear(3136, self.n_features)
        self._h5 = nn.Linear(self.n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h5.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None):
        h = F.relu(self._h1(state.float() / 255.))
        h = F.relu(self._h2(h))
        h = F.relu(self._h3(h))
        h = F.relu(self._h4(h.view(-1, 3136)))
        q = self._h5(h)

        if action is None:
            return q
        else:
            return torch.squeeze(q.gather(1, action.long()))


class AtariFeatureNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[0]

        self._h1 = nn.Conv2d(n_input, 32, kernel_size=8, stride=4)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self._h4 = nn.Linear(3136, AtariNetwork.n_features)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, state, action=None):
        h = F.relu(self._h1(state.float() / 255.))
        h = F.relu(self._h2(h))
        h = F.relu(self._h3(h))
        return F.relu(self._h4(h.view(-1, 3136)))
