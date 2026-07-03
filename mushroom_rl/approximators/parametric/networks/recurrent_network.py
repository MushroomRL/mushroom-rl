import torch
import torch.nn as nn

from mushroom_rl.utils.torch_utils import TorchUtils


class RecurrentActorNetwork(nn.Module):
    """
    Recurrent actor network returning both the action mean and the next hidden state, so
    ``output_shape`` must be ``[action_shape, policy_state_shape]``.

    """
    def __init__(self, input_shape, output_shape, n_features, dim_env_state, rnn_type,
                 n_hidden_features, num_hidden_layers=1, **kwargs):
        super().__init__()

        assert isinstance(output_shape, list) and len(output_shape) == 2, \
            'RecurrentActorNetwork requires output_shape=[action_shape, policy_state_shape].'

        dim_action = output_shape[0][0]
        self._dim_env_state = dim_env_state
        self._num_hidden_layers = num_hidden_layers
        self._n_hidden_features = n_hidden_features

        rnn = TorchUtils.get_recurrent_network(rnn_type)

        self._h1_o = nn.Linear(dim_env_state, n_features)
        self._h1_o_post_rnn = nn.Linear(dim_env_state, n_features)

        self._rnn = rnn(input_size=n_features,
                        hidden_size=n_hidden_features,
                        num_layers=num_hidden_layers,
                        batch_first=True)

        self._h3 = nn.Linear(n_hidden_features + n_features, dim_action)
        self._act_func = nn.ReLU()
        self._tanh = nn.Tanh()

        nn.init.xavier_uniform_(self._h1_o.weight, gain=nn.init.calculate_gain('relu') * 0.05)
        nn.init.xavier_uniform_(self._h1_o_post_rnn.weight, gain=nn.init.calculate_gain('relu') * 0.05)
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain('relu') * 0.05)

    def forward(self, state, policy_state, lengths):
        input_rnn = self._act_func(self._h1_o(state))

        packed_seq = nn.utils.rnn.pack_padded_sequence(input_rnn, lengths, enforce_sorted=False,
                                                        batch_first=True)

        policy_state_reshaped = policy_state.view(-1, self._num_hidden_layers, self._n_hidden_features)
        policy_state_reshaped = torch.swapaxes(policy_state_reshaped, 0, 1)

        out_rnn, next_hidden = self._rnn(packed_seq, policy_state_reshaped)

        features_rnn, _ = nn.utils.rnn.pad_packed_sequence(out_rnn, batch_first=True)
        rel_indices = lengths.view(-1, 1, 1) - 1
        features_rnn = torch.squeeze(torch.take_along_dim(features_rnn, rel_indices, dim=1), dim=1)

        last_state = torch.squeeze(torch.take_along_dim(state, rel_indices, dim=1), dim=1)
        feature_sa = self._act_func(self._h1_o_post_rnn(last_state))

        input_last_layer = torch.concat([feature_sa, features_rnn], dim=1)
        a = self._h3(input_last_layer)

        return a, torch.swapaxes(next_hidden, 0, 1)


class RecurrentCriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, dim_env_state, dim_action, rnn_type,
                 n_hidden_features=128, n_features=128, num_hidden_layers=1,
                 hidden_state_treatment='zero_initial', **kwargs):
        super().__init__()

        assert hidden_state_treatment in ['zero_initial', 'use_policy_hidden_state']

        self._dim_env_state = dim_env_state
        self._dim_action = dim_action
        self._use_policy_hidden_states = hidden_state_treatment == 'use_policy_hidden_state'
        self._num_hidden_layers = num_hidden_layers
        self._n_hidden_features = n_hidden_features

        rnn = TorchUtils.get_recurrent_network(rnn_type)

        self._h1_o = nn.Linear(dim_env_state, n_features)
        self._h1_o_post_rnn = nn.Linear(dim_env_state, n_features)

        self._rnn = rnn(input_size=n_features,
                        hidden_size=n_hidden_features,
                        num_layers=num_hidden_layers,
                        batch_first=True)

        self._hq_1 = nn.Linear(n_hidden_features + n_features, n_features)
        self._hq_2 = nn.Linear(n_features, 1)
        self._act_func = nn.ReLU()

        nn.init.xavier_uniform_(self._h1_o.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h1_o_post_rnn.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._hq_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._hq_2.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, state, policy_state, lengths):
        input_rnn = self._act_func(self._h1_o(state))

        packed_seq = nn.utils.rnn.pack_padded_sequence(input_rnn, lengths, enforce_sorted=False,
                                                        batch_first=True)
        if self._use_policy_hidden_states:
            policy_state_reshaped = policy_state.view(-1, self._num_hidden_layers, self._n_hidden_features)
            policy_state_reshaped = torch.swapaxes(policy_state_reshaped, 0, 1)
            out_rnn, _ = self._rnn(packed_seq, policy_state_reshaped)
        else:
            out_rnn, _ = self._rnn(packed_seq)

        features_rnn, _ = nn.utils.rnn.pad_packed_sequence(out_rnn, batch_first=True)
        rel_indices = lengths.view(-1, 1, 1) - 1
        features_rnn = torch.squeeze(torch.take_along_dim(features_rnn, rel_indices, dim=1), dim=1)

        last_state = torch.squeeze(torch.take_along_dim(state, rel_indices, dim=1), dim=1)
        feature_s = self._act_func(self._h1_o_post_rnn(last_state))

        input_last_layer = torch.concat([feature_s, features_rnn], dim=1)
        q = self._hq_2(self._act_func(self._hq_1(input_last_layer)))

        return torch.squeeze(q)
