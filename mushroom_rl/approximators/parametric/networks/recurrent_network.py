import math

import torch
import torch.nn as nn

from mushroom_rl.utils.torch_utils import TorchUtils


class RecurrentNetwork(nn.Module):
    """
    Base class for the recurrent networks. This class holds the conversions between the policy state carried by
    the agent and the hidden state the recurrent layer expects.

    An LSTM layer carries both a hidden and a cell state, so its policy state is shaped
    ``(2, num_hidden_layers, n_hidden_features)``; every other layer type carries only a hidden state and drops the
    leading axis, giving ``(num_hidden_layers, n_hidden_features)``.

    A subclass whose ``forward`` returns the next policy state alongside its main output sets
    ``returns_policy_state`` to ``True``, so that the approximator can add it to the output shapes.

    """
    returns_policy_state = False

    @classmethod
    def resolve_output_shape(cls, output_shape, **params):
        """
        Complete the output shapes requested by the caller with the ones the network adds by itself. A network
        returning the next policy state has it appended, so that the approximator knows the network returns
        two tensors. An ``output_shape`` already given as a list of shapes is left untouched, letting the
        caller declare the outputs explicitly.

        Args:
            output_shape (tuple, list): the shape of the main output of the network, or the list of the
                shapes of all its outputs;
            **params: the parameters the network is constructed with.

        Returns:
            The shape, or list of shapes, of the outputs the network returns.

        """
        if cls.returns_policy_state and not isinstance(output_shape, list):
            output_shape = [output_shape, cls.compute_policy_state_shape(**params)]

        return output_shape

    @staticmethod
    def compute_policy_state_shape(rnn_type, n_hidden_features, num_hidden_layers=1, **kwargs):
        """
        Compute the shape of the policy state the network carries, without building the network, so that
        it can be declared before construction.

        Args:
            rnn_type (str): type of recurrent layer, see ``TorchUtils.get_recurrent_network``;
            n_hidden_features (int): size of the recurrent layer's hidden state;
            num_hidden_layers (int, 1): number of stacked recurrent layers;
            **kwargs: other network parameters (unused).

        Returns:
            The shape of a single policy state, as a tuple.

        """
        if TorchUtils.get_recurrent_network(rnn_type) is nn.LSTM:
            return 2, num_hidden_layers, n_hidden_features
        else:
            return num_hidden_layers, n_hidden_features

    @staticmethod
    def _split_policy_state(policy_state):
        """
        Turn a batch of policy states into the hidden state the recurrent layer expects, i.e. one
        ``(num_layers, batch, n_hidden_features)`` tensor per hidden state, made contiguous as cuDNN
        requires.

        Args:
            policy_state (torch.Tensor): a ``(batch, *policy_state_shape)`` tensor.

        Returns:
            The hidden state of the recurrent layer, a tuple of tensors for LSTM layers.

        """
        if policy_state.ndim == 4:
            return tuple(policy_state.permute(1, 2, 0, 3).contiguous())
        else:
            return policy_state.permute(1, 0, 2).contiguous()

    @staticmethod
    def _merge_policy_state(hidden_state):
        """
        Turn the hidden state returned by a recurrent layer back into a batch of policy states.

        Args:
            hidden_state (torch.Tensor, tuple): the hidden state returned by the recurrent layer, a
                ``(h, c)`` tuple for LSTM layers.

        Returns:
            The ``(batch, *policy_state_shape)`` tensor of policy states.

        """
        if isinstance(hidden_state, tuple):
            return torch.stack(hidden_state).permute(2, 0, 1, 3)
        else:
            return hidden_state.permute(1, 0, 2)

    @staticmethod
    def _last_hidden_state(hidden_state):
        """
        Extract the final hidden state of the last recurrent layer. Fed a packed sequence, the recurrent
        layer stops each sequence at its own length, so this is already the output of each sequence at its
        last valid timestep and does not have to be gathered out of the padded output.

        Args:
            hidden_state (torch.Tensor, tuple): the hidden state returned by the recurrent layer, a
                ``(h, c)`` tuple for LSTM layers.

        Returns:
            The ``(batch, feature)`` tensor of the final hidden state of the last layer.

        """
        h = hidden_state[0] if isinstance(hidden_state, tuple) else hidden_state

        return h[-1]

    @staticmethod
    def _select_by_length(x, lengths):
        """
        Select, for every element of the batch, the entry of the sequence at its own length.

        Args:
            x (torch.Tensor): a ``(batch, sequence, feature)`` tensor;
            lengths (torch.Tensor): the length of each sequence, kept on the CPU as
                ``pack_padded_sequence`` requires.

        Returns:
            The ``(batch, feature)`` tensor of the selected entries.

        """
        if x.shape[1] > 1:
            rel_indices = (lengths.view(-1, 1, 1) - 1).to(x.device)
            return torch.squeeze(torch.take_along_dim(x, rel_indices, dim=1), dim=1)
        else:
            # a single timestep leaves index 0 as the only valid one, so the gather is just a squeeze
            return torch.squeeze(x, dim=1)


class RecurrentActorNetwork(RecurrentNetwork):
    """
    Recurrent actor network returning both the action mean and the next hidden state.

    """
    returns_policy_state = True

    def __init__(self, input_shape, output_shape, n_features, rnn_type,
                 n_hidden_features, num_hidden_layers=1, action_history_shape=None, **kwargs):
        """
        Constructor.

        Args:
            input_shape (tuple): per-timestep shape of the input (the observation);
            output_shape (list): the shapes of the outputs of the network;
            n_features (int): size of the layers feeding into and out of the recurrent layer;
            rnn_type (str): type of recurrent layer, see ``TorchUtils.get_recurrent_network``;
            n_hidden_features (int): size of the recurrent layer's hidden state;
            num_hidden_layers (int, 1): number of stacked recurrent layers;
            action_history_shape (tuple, None): per-timestep shape of the previous-action input; when set,
                the flattened previous action is concatenated to the observation of each timestep;
            **kwargs: other parameters (unused).

        """
        super().__init__()

        assert isinstance(output_shape, list) and len(output_shape) == 2, \
            'RecurrentActorNetwork requires output_shape=[action_shape, policy_state_shape].'

        dim_env_state = math.prod(input_shape)
        dim_action = output_shape[0][0]
        self._use_action_history = action_history_shape is not None

        # when the previous action is fed in, it is concatenated to the observation of each timestep
        dim_input = dim_env_state + (math.prod(action_history_shape) if self._use_action_history else 0)

        rnn = TorchUtils.get_recurrent_network(rnn_type)

        self._h1_o = nn.Linear(dim_input, n_features)
        self._h1_o_post_rnn = nn.Linear(dim_input, n_features)

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

    def forward(self, state, policy_state, lengths, action_history=None, **kwargs):
        state = state.flatten(start_dim=2)
        net_input = torch.concat([state, action_history.flatten(start_dim=2)], dim=-1) \
            if self._use_action_history else state

        input_rnn = self._act_func(self._h1_o(net_input))

        packed_seq = nn.utils.rnn.pack_padded_sequence(input_rnn, lengths, enforce_sorted=False,
                                                       batch_first=True)

        _, next_hidden = self._rnn(packed_seq, self._split_policy_state(policy_state))

        features_rnn = self._last_hidden_state(next_hidden)

        last_input = self._select_by_length(net_input, lengths)
        feature_sa = self._act_func(self._h1_o_post_rnn(last_input))

        input_last_layer = torch.concat([feature_sa, features_rnn], dim=1)
        a = self._h3(input_last_layer)

        return a, self._merge_policy_state(next_hidden)


class RecurrentCriticNetwork(RecurrentNetwork):
    """
    Recurrent critic network, returning the value function of the input sequence.

    """
    def __init__(self, input_shape, output_shape, rnn_type,
                 n_hidden_features=128, n_features=128, num_hidden_layers=1,
                 hidden_state_treatment='zero_initial', action_history_shape=None, **kwargs):
        """
        Constructor.

        Args:
            input_shape (tuple): per-timestep shape of the input (the observation);
            output_shape (tuple): shape of the output (the value function);
            rnn_type (str): type of recurrent layer, see ``TorchUtils.get_recurrent_network``;
            n_hidden_features (int, 128): size of the recurrent layer's hidden state;
            n_features (int, 128): size of the layers feeding into and out of the recurrent layer;
            num_hidden_layers (int, 1): number of stacked recurrent layers;
            hidden_state_treatment (str, 'zero_initial'): either ``'zero_initial'``, to start the
                recurrent layer from a zero hidden state, or ``'use_policy_hidden_state'``, to start it
                from the policy's hidden state instead;
            action_history_shape (tuple, None): per-timestep shape of the previous-action input; when set,
                the flattened previous action is concatenated to the observation of each timestep;
            **kwargs: other parameters (unused).

        """
        super().__init__()

        assert hidden_state_treatment in ['zero_initial', 'use_policy_hidden_state']

        dim_env_state = math.prod(input_shape)
        self._use_policy_hidden_states = hidden_state_treatment == 'use_policy_hidden_state'
        self._use_action_history = action_history_shape is not None

        # when the previous action is fed in, it is concatenated to the observation of each timestep
        dim_input = dim_env_state + (math.prod(action_history_shape) if self._use_action_history else 0)

        rnn = TorchUtils.get_recurrent_network(rnn_type)

        self._h1_o = nn.Linear(dim_input, n_features)
        self._h1_o_post_rnn = nn.Linear(dim_input, n_features)

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

    def forward(self, state, policy_state, lengths, action_history=None, **kwargs):
        state = state.flatten(start_dim=2)
        net_input = torch.concat([state, action_history.flatten(start_dim=2)], dim=-1) \
            if self._use_action_history else state

        input_rnn = self._act_func(self._h1_o(net_input))

        packed_seq = nn.utils.rnn.pack_padded_sequence(input_rnn, lengths, enforce_sorted=False,
                                                       batch_first=True)
        if self._use_policy_hidden_states:
            _, next_hidden = self._rnn(packed_seq, self._split_policy_state(policy_state))
        else:
            _, next_hidden = self._rnn(packed_seq)

        features_rnn = self._last_hidden_state(next_hidden)

        last_input = self._select_by_length(net_input, lengths)
        feature_s = self._act_func(self._h1_o_post_rnn(last_input))

        input_last_layer = torch.concat([feature_s, features_rnn], dim=1)
        q = self._hq_2(self._act_func(self._hq_1(input_last_layer)))

        return torch.squeeze(q)
