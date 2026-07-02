from mushroom_rl.rl_utils.replay_memory.replay_memory import ReplayMemory


class SequenceReplayMemory(ReplayMemory):
    """
    This class extend the base replay memory to allow sampling sequences of a certain length. This is useful for
    training recurrent agents or agents operating on a window of states etc.

    """
    def __init__(self, mdp_info, agent_info, initial_size, max_size, truncation_length):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            agent_info (AgentInfo): information about the agent;
            initial_size (int): initial size of the replay buffer;
            max_size (int): maximum size of the replay buffer;
            truncation_length (int): truncation length to be sampled;
        """
        self._truncation_length = truncation_length
        self._action_space_shape = mdp_info.action_space.shape

        super(SequenceReplayMemory, self).__init__(mdp_info, agent_info, initial_size, max_size,
                                                   store_policy_state=True)

        self._add_save_attr(
            _truncation_length='primitive',
            _action_space_shape='primitive'
        )

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.

        Args:
            n_samples (int): the number of samples to return.

        Returns:
            The requested number of samples.

        """
        backend = self._dataset.array_backend

        s = backend.zeros(n_samples, self._truncation_length, *self._mdp_info.observation_space.shape,
                          dtype=self._mdp_info.observation_space.data_type)
        a = backend.zeros(n_samples, self._truncation_length, *self._mdp_info.action_space.shape,
                          dtype=self._mdp_info.action_space.data_type)
        r = backend.zeros(n_samples, 1)
        ss = backend.zeros(n_samples, self._truncation_length, *self._mdp_info.observation_space.shape,
                           dtype=self._mdp_info.observation_space.data_type)
        ab = backend.zeros(n_samples, 1, dtype=int)
        last = backend.zeros(n_samples, dtype=int)
        ps = backend.zeros(n_samples, self._truncation_length, *self._agent_info.policy_state_shape)
        nps = backend.zeros(n_samples, self._truncation_length, *self._agent_info.policy_state_shape)
        pa = backend.zeros(n_samples, self._truncation_length, *self._mdp_info.action_space.shape,
                           dtype=self._mdp_info.action_space.data_type)
        lengths = list()

        for num, i in enumerate(backend.randint(0, self.size, (n_samples,))):
            i = int(i)
            begin_seq = max(i - self._truncation_length + 1, 0)
            end_seq = i + 1

            lasts_absorbing = self._dataset.last[begin_seq: i]
            begin_traj = backend.where(lasts_absorbing > 0)
            more_than_one_traj = len(*begin_traj) > 0
            if more_than_one_traj:
                begin_seq = begin_seq + begin_traj[0][-1] + 1

            if more_than_one_traj or begin_seq == 0 or self._dataset.last[begin_seq - 1]:
                prev_actions = self._dataset.action[begin_seq:end_seq - 1]
                init_prev_action = backend.zeros(1, *self._action_space_shape)
                if len(prev_actions) == 0:
                    prev_actions = init_prev_action
                else:
                    prev_actions = backend.concatenate([init_prev_action, prev_actions])
            else:
                prev_actions = self._dataset.action[begin_seq - 1:end_seq - 1]

            s[num, :end_seq - begin_seq] = self._dataset.state[begin_seq:end_seq]
            ss[num, :end_seq - begin_seq] = self._dataset.next_state[begin_seq:end_seq]
            a[num, :end_seq - begin_seq] = self._dataset.action[begin_seq:end_seq]
            ps[num, :end_seq - begin_seq] = self._dataset.policy_state[begin_seq:end_seq]
            nps[num, :end_seq - begin_seq] = self._dataset.policy_next_state[begin_seq:end_seq]
            pa[num, :end_seq - begin_seq] = prev_actions
            r[num] = self._dataset.reward[i]
            ab[num] = self._dataset.absorbing[i]
            last[num] = self._dataset.last[i]

            lengths.append(end_seq - begin_seq)

        if self._dataset.is_stateful:
            return s, a, r, ss, ab, last, ps, nps, pa, lengths
        else:
            return s, a, r, ss, ab, last, pa, lengths
