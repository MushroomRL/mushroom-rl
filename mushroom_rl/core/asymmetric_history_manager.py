from numbers import Integral

from mushroom_rl.core.history_manager import HistoryManager


class AsymmetricHistoryManager(HistoryManager):
    """
    History manager building the policy input and a critic input from different entries of the preprocessed
    observation, each with its own history length. The policy input is the one of
    :class:`~mushroom_rl.core.history_manager.HistoryManager`, restricted to the actor entries. The critic input is
    built only when parsing a dataset, into the ``critic_state`` and ``critic_next_state`` of the returned batches: the
    critic entries, stacked over the critic history length.

    """
    def __init__(self, mdp_info, agent_info, streams=None, preprocessors=None, actor_indices=None,
                 critic_indices=None):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information about the MDP, providing the observation space the indices select from;
            agent_info (AgentInfo): information about the agent, providing the array backend and the device on which
                the manager keeps its buffers;
            streams (dict, None): the named streams assembled by the manager, as in
                :class:`~mushroom_rl.core.history_manager.HistoryManager`; the ``critic_history`` stream, when given,
                stacks the critic entries of the observation and is never returned to the policy;
            preprocessors (list, None): preprocessors applied to every observation before its entries are selected;
            actor_indices (list, None): the entries of the observation the actor reads; all of them when ``None``;
            critic_indices (list, None): the entries of the observation the critic reads; all of them when ``None``.

        Raises:
            AssertionError: if indices are given for an observation that is not a vector, or are empty, not integers,
                out of range or repeated, or if a reserved stream does not match the entries it stacks.

        """
        self._actor_indices = self._validated_indices(actor_indices, mdp_info, 'actor_indices')
        self._critic_indices = self._validated_indices(critic_indices, mdp_info, 'critic_indices')

        super().__init__(mdp_info, agent_info, streams=streams, preprocessors=preprocessors)

        self._actor_selection = self._build_selection(self._actor_indices)
        self._critic_selection = self._build_selection(self._critic_indices)

        self._add_save_attr(
            _actor_indices='primitive',
            _critic_indices='primitive',
            _actor_selection='none',
            _critic_selection='none'
        )

    def parse_initial_critic_state(self, dataset, to=None):
        """
        Build the critic input of each episode start of a dataset, i.e. the critic counterpart of
        :meth:`~mushroom_rl.core.history_manager.HistoryManager.parse_initial_state`. These windows have no history
        behind them, so the older entries are zero-padded.

        Args:
            dataset (Dataset): the dataset to parse;
            to (str, None): the backend of the returned array; when ``None`` the agent backend is used.

        Returns:
            The critic input of the initial states.

        """
        states = self._backend.convert(dataset.get_init_states(), device=self._device)
        critic_state = self._select(self.preprocess(states), self._critic_selection)

        if 'critic_history' in self._stream_specs:
            last = self._backend.ones(len(critic_state), dtype=bool, device=self._device)
            critic_state = self._stack_history('critic_history', critic_state, last)

        return self._convert_output(to, critic_state)

    @classmethod
    def default_streams(cls, mdp_info, agent_info, history_length=None, action_history_length=None, actor_indices=None,
                        critic_indices=None, critic_history_length=1):
        """
        Build a manager wired up with the default reserved streams, as
        :meth:`~mushroom_rl.core.history_manager.HistoryManager.default_streams`, reading the actor and critic entries
        of the observation.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            agent_info (AgentInfo): information about the agent;
            history_length (int, None): number of observations stacked as actor input;
            action_history_length (int, None): number of previous actions stacked as policy input;
            actor_indices (list, None): the entries of the observation the actor reads; all of them when ``None``;
            critic_indices (list, None): the entries of the observation the critic reads; all of them when ``None``;
            critic_history_length (int, 1): number of observations stacked as critic input.

        Returns:
            The :class:`AsymmetricHistoryManager` instance.

        Raises:
            AssertionError: if indices are given for an observation that is not a vector, or are empty, not integers,
                out of range or repeated.

        """
        streams = cls._default_specs(mdp_info, history_length, action_history_length)
        if actor_indices is not None and 'obs_history' in streams:
            streams['obs_history']['shape'] = (len(actor_indices),)
        if critic_history_length > 1:
            shape = mdp_info.observation_space.shape if critic_indices is None else (len(critic_indices),)
            streams['critic_history'] = dict(length=critic_history_length, shape=shape,
                                             dtype=mdp_info.observation_space.data_type)

        return cls(mdp_info, agent_info, streams=streams, actor_indices=actor_indices, critic_indices=critic_indices)

    @property
    def actor_indices(self):
        """
        The entries of the observation the actor reads, or ``None`` when it reads all of them.

        """
        return self._actor_indices

    @property
    def critic_indices(self):
        """
        The entries of the observation the critic reads, or ``None`` when it reads all of them.

        """
        return self._critic_indices

    @property
    def critic_history_length(self):
        """
        The number of observations stacked as critic input.

        """
        return self._stream_specs['critic_history']['length'] if 'critic_history' in self._stream_specs else 1

    def _reserved_entries(self, mdp_info):
        entries = super()._reserved_entries(mdp_info)
        dtype = mdp_info.observation_space.data_type
        entries.update(obs_history=(self._selected_shape(mdp_info, self._actor_indices), dtype),
                       critic_history=(self._selected_shape(mdp_info, self._critic_indices), dtype))
        return entries

    def _observation_streams(self):
        streams = super()._observation_streams()
        streams.update(critic_history=self._critic_selection)
        return streams

    def _build_selection(self, indices):
        if indices is None:
            return None
        if indices == list(range(indices[0], indices[0] + len(indices))):
            return slice(indices[0], indices[0] + len(indices))
        return self._backend.from_list(indices, device=self._device)

    def _post_load(self, device):
        super()._post_load(device)

        self._actor_selection = self._build_selection(self._actor_indices)
        self._critic_selection = self._build_selection(self._critic_indices)

    @staticmethod
    def _validated_indices(indices, mdp_info, name):
        if indices is None:
            return None
        shape = mdp_info.observation_space.shape
        values = indices.tolist() if hasattr(indices, 'tolist') else list(indices)
        assert len(shape) == 1, f"{name}: entries can be selected only from a vector observation, got shape {shape}."
        assert len(values) > 0, f"{name}: no observation entry is selected."
        assert all(isinstance(value, Integral) and not isinstance(value, bool) for value in values), \
            f"{name}: the indices must be integers, got {values}."
        assert all(0 <= value < shape[0] for value in values), \
            f"{name}: the indices must lie in [0, {shape[0]}), got {values}."
        assert len(set(values)) == len(values), f"{name}: an index is repeated in {values}."
        return [int(value) for value in values]

    @staticmethod
    def _selected_shape(mdp_info, indices):
        return mdp_info.observation_space.shape if indices is None else (len(indices),)
