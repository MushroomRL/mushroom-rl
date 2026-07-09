import numpy as np
import pytest
import torch

from mushroom_rl.core import MDPInfo, AgentInfo, Dataset
from mushroom_rl.core.spaces import Box, Discrete
from mushroom_rl.core.history_manager import HistoryManager
from mushroom_rl.rl_utils.replay_memory import ReplayMemory, SequenceReplayMemory, PrioritizedReplayMemory
from mushroom_rl.rl_utils.parameters import LinearParameter


def make_mdp_info(obs_shape, act_shape, backend='numpy', gamma=0.99):
    obs_space = Box(np.full(obs_shape, -1.0), np.full(obs_shape, 1.0), obs_shape)
    act_space = Box(np.full(act_shape, -1.0), np.full(act_shape, 1.0), act_shape)
    return MDPInfo(obs_space, act_space, gamma=gamma, horizon=100, backend=backend)


def make_agent_info(policy_state_shape=None, backend='numpy'):
    return AgentInfo(is_episodic=False, policy_state_shape=policy_state_shape, backend=backend)


def make_dataset(n, rng, obs_shape, act_shape, policy_state_shape=None, backend='numpy', episode_ends=None):
    states = rng.randn(n, *obs_shape)
    actions = rng.randn(n, *act_shape)
    rewards = rng.randn(n)
    next_states = rng.randn(n, *obs_shape)
    absorbings = np.zeros(n)
    lasts = np.zeros(n)
    if episode_ends is None:
        lasts[-1] = 1.0
    else:
        for idx in episode_ends:
            lasts[idx] = 1.0

    policy_state = policy_next_state = None
    if policy_state_shape is not None:
        policy_state = rng.randn(n, *policy_state_shape)
        policy_next_state = rng.randn(n, *policy_state_shape)

    return (Dataset.from_array(states, actions, rewards, next_states, absorbings, lasts,
                               policy_state=policy_state, policy_next_state=policy_next_state,
                               backend=backend),
            states, actions, rewards, next_states, policy_state, policy_next_state)


def test_replay_memory_add_get():
    obs_shape = (4,)
    act_shape = (2,)
    n = 10

    rng = np.random.RandomState(42)
    dataset, states, actions, rewards, next_states, _, _ = make_dataset(
        n, rng, obs_shape, act_shape)

    mdp_info = make_mdp_info(obs_shape, act_shape)
    agent_info = make_agent_info()
    rm = ReplayMemory(mdp_info, agent_info, initial_size=5, max_size=50)
    rm.add(dataset)

    assert rm.size == 10
    assert rm.initialized

    np.random.seed(7)
    s, a, r, ss, ab, last = rm.get(4)

    expected_idxs = np.array([4, 9, 6, 3])
    s_test = np.array([[-1.01283112,  0.31424733, -0.90802408, -1.4123037],
                       [0.2088636,  -1.95967012, -1.32818605,  0.19686124],
                       [-0.54438272,  0.11092259, -1.15099358,  0.37569802],
                       [0.24196227, -1.91328024, -1.72491783, -0.56228753]])
    r_test = np.array([0.81252582, -0.64511975, -0.07201012, -1.19620662])

    assert np.allclose(s, states[expected_idxs])
    assert np.allclose(a, actions[expected_idxs])
    assert np.allclose(r, rewards[expected_idxs])
    assert np.allclose(ss, next_states[expected_idxs])
    assert np.allclose(s, s_test)
    assert np.allclose(r, r_test)


def test_replay_memory_not_initialized():
    obs_shape = (4,)
    act_shape = (2,)

    rng = np.random.RandomState(0)
    dataset, *_ = make_dataset(10, rng, obs_shape, act_shape)

    mdp_info = make_mdp_info(obs_shape, act_shape)
    agent_info = make_agent_info()
    rm = ReplayMemory(mdp_info, agent_info, initial_size=20, max_size=100)
    rm.add(dataset)

    assert rm.size == 10
    assert not rm.initialized


def test_replay_memory_initialized_at_boundary():
    obs_shape = (4,)
    act_shape = (2,)

    rng = np.random.RandomState(0)
    dataset, *_ = make_dataset(10, rng, obs_shape, act_shape)

    mdp_info = make_mdp_info(obs_shape, act_shape)
    agent_info = make_agent_info()
    rm = ReplayMemory(mdp_info, agent_info, initial_size=10, max_size=100)
    rm.add(dataset)

    assert rm.size == 10
    assert rm.initialized


def test_replay_memory_wrapping_when_full():
    obs_shape = (4,)
    act_shape = (2,)
    max_size = 15
    n = 10

    rng = np.random.RandomState(42)
    ds_a, states_a, _, rewards_a, next_states_a, _, _ = make_dataset(n, rng, obs_shape, act_shape)
    ds_b, states_b, _, rewards_b, next_states_b, _, _ = make_dataset(n, rng, obs_shape, act_shape)

    mdp_info = make_mdp_info(obs_shape, act_shape)
    agent_info = make_agent_info()
    rm = ReplayMemory(mdp_info, agent_info, initial_size=5, max_size=max_size)
    rm.add(ds_a)
    rm.add(ds_b)

    assert rm._full
    assert rm.size == max_size

    assert np.allclose(rm._dataset.state[:5], states_b[5:10])
    assert np.allclose(rm._dataset.state[5:10], states_a[5:10])
    assert np.allclose(rm._dataset.state[10:15], states_b[0:5])
    assert np.allclose(rm._dataset.reward[:5], rewards_b[5:10])
    assert np.allclose(rm._dataset.next_state[10:15], next_states_b[0:5])


def test_replay_memory_n_steps_return():
    obs_shape = (4,)
    act_shape = (2,)
    n, n_steps, gamma = 10, 3, 0.9

    rng = np.random.RandomState(0)
    states = rng.randn(n, *obs_shape)
    actions = rng.randn(n, *act_shape)
    rewards = np.ones(n)
    next_states = rng.randn(n, *obs_shape)
    absorbings = np.zeros(n)
    lasts = np.zeros(n)
    lasts[-1] = 1.0
    dataset = Dataset.from_array(states, actions, rewards, next_states, absorbings, lasts)

    mdp_info = make_mdp_info(obs_shape, act_shape, gamma=gamma)
    agent_info = make_agent_info()
    rm = ReplayMemory(mdp_info, agent_info, initial_size=1, max_size=50, n_steps_return=n_steps)
    rm.add(dataset)

    assert rm.size == n - n_steps + 1

    expected_reward = 1.0 + gamma + gamma ** 2
    assert np.allclose(rm._dataset.reward[:rm.size], expected_reward)

    assert np.allclose(rm._dataset.next_state[0], next_states[n_steps - 1])
    assert np.allclose(rm._dataset.next_state[rm.size - 1], next_states[n - 1])


def test_replay_memory_stateful():
    obs_shape = (4,)
    act_shape = (2,)
    policy_state_shape = (8,)
    n = 10

    rng = np.random.RandomState(5)
    dataset, states, actions, rewards, _, policy_states, _ = make_dataset(
        n, rng, obs_shape, act_shape, policy_state_shape=policy_state_shape)

    mdp_info = make_mdp_info(obs_shape, act_shape)
    agent_info = make_agent_info(policy_state_shape=policy_state_shape)
    rm = ReplayMemory(mdp_info, agent_info, initial_size=5, max_size=50, store_policy_state=True)

    assert rm._dataset.is_stateful

    rm.add(dataset)

    np.random.seed(7)
    s, a, r, ss, ab, last, ps, nps = rm.get(4)

    expected_idxs = np.array([4, 9, 6, 3])
    ps_test = np.array([[1.21228341, -0.6346525, -1.5996985, 0.87715281, -0.09383245,
                         -0.05567103, -0.88942073, -1.30095145],
                        [-0.14312642, -0.22418983, -1.03849524, -0.17170905, 0.47634618,
                         -0.41417827, -1.26408334, -0.57321556],
                        [-1.27962318, 0.03654264, -0.64635659, 0.54856784, 0.21054246,
                         0.34650175, -0.56705117, 0.41367881],
                        [0.92751621, 1.63995407, 2.07361553, 0.70979786, 0.74715259,
                         1.46309548, 1.73844881, 1.46520488]])

    assert np.allclose(ps, policy_states[expected_idxs])
    assert np.allclose(ps, ps_test)
    assert np.allclose(s, states[expected_idxs])


def test_replay_memory_reset():
    obs_shape = (4,)
    act_shape = (2,)

    rng = np.random.RandomState(1)
    dataset, *_ = make_dataset(10, rng, obs_shape, act_shape)

    mdp_info = make_mdp_info(obs_shape, act_shape)
    agent_info = make_agent_info()
    rm = ReplayMemory(mdp_info, agent_info, initial_size=1, max_size=20)
    rm.add(dataset)

    assert rm.size == 10

    rm.reset()

    assert rm.size == 0
    assert not rm._full
    assert rm._idx == 0


def test_replay_memory_torch_backend():
    obs_shape = (4,)
    act_shape = (2,)
    n = 10

    rng = np.random.RandomState(42)
    dataset, states, actions, rewards, next_states, _, _ = make_dataset(
        n, rng, obs_shape, act_shape, backend='torch')

    mdp_info = make_mdp_info(obs_shape, act_shape, backend='torch')
    agent_info = make_agent_info(backend='torch')
    rm = ReplayMemory(mdp_info, agent_info, initial_size=5, max_size=50)
    rm.add(dataset)

    assert rm.size == 10

    torch.manual_seed(7)
    s, a, r, ss, ab, last = rm.get(4)

    assert isinstance(s, torch.Tensor)

    torch.manual_seed(7)
    idxs = torch.randint(0, 10, (4,)).numpy()
    assert np.allclose(s.numpy(), states[idxs])
    assert np.allclose(r.numpy(), rewards[idxs])


def test_sequence_replay_memory_shapes_and_values():
    obs_shape = (4,)
    act_shape = (2,)
    policy_state_shape = (8,)
    truncation_length = 5
    n = 30

    rng = np.random.RandomState(42)
    dataset, states, actions, rewards, _, policy_states, _ = make_dataset(
        n, rng, obs_shape, act_shape, policy_state_shape=policy_state_shape,
        episode_ends=[14, 29])

    mdp_info = make_mdp_info(obs_shape, act_shape)
    agent_info = make_agent_info(policy_state_shape=policy_state_shape)
    rm = SequenceReplayMemory(mdp_info, agent_info, initial_size=10, max_size=100,
                              truncation_length=truncation_length)
    rm.add(dataset)

    n_samples = 2
    np.random.seed(3)
    s_seq, a_seq, r_seq, ss_seq, ab_seq, last_seq, ps_seq, nps_seq, lengths = rm.get(n_samples)

    expected_idxs = np.array([10, 24])
    expected_lengths = [5, 5]

    assert s_seq.shape == (n_samples, truncation_length, *obs_shape)
    assert a_seq.shape == (n_samples, truncation_length, *act_shape)
    assert r_seq.shape == (n_samples, 1)
    assert ps_seq.shape == (n_samples, truncation_length, *policy_state_shape)
    assert lengths == expected_lengths

    assert np.allclose(r_seq[:, 0], rm._dataset.reward[expected_idxs])

    for i, (idx, length) in enumerate(zip(expected_idxs, expected_lengths)):
        begin = idx - length + 1
        assert np.allclose(s_seq[i, :length], rm._dataset.state[begin:idx + 1])
        assert np.allclose(ps_seq[i, :length], rm._dataset.policy_state[begin:idx + 1])


def test_sequence_replay_memory_episode_boundary():
    obs_shape = (4,)
    act_shape = (2,)
    policy_state_shape = (8,)
    episode_length = 5

    rng = np.random.RandomState(0)
    dataset, *_ = make_dataset(10, rng, obs_shape, act_shape, policy_state_shape=policy_state_shape,
                               episode_ends=[4, 9])

    mdp_info = make_mdp_info(obs_shape, act_shape)
    agent_info = make_agent_info(policy_state_shape=policy_state_shape)
    rm = SequenceReplayMemory(mdp_info, agent_info, initial_size=5, max_size=100,
                              truncation_length=10)
    rm.add(dataset)

    for seed in range(20):
        np.random.seed(seed)
        *_, lengths = rm.get(1)
        assert lengths[0] <= episode_length


def test_sequence_replay_memory_history_composition():
    obs_shape = (4,)
    act_shape = (2,)
    policy_state_shape = (8,)
    truncation_length = 3
    history_length = 2
    n = 15
    n_samples = 3

    rng = np.random.RandomState(1)
    dataset, *_ = make_dataset(n, rng, obs_shape, act_shape,
                               policy_state_shape=policy_state_shape, episode_ends=[14])

    mdp_info = make_mdp_info(obs_shape, act_shape)
    agent_info = make_agent_info(policy_state_shape=policy_state_shape)
    history_manager = HistoryManager.from_infos(mdp_info, agent_info, history_length=history_length)
    rm = SequenceReplayMemory(mdp_info, agent_info, initial_size=5, max_size=100,
                              truncation_length=truncation_length, history_manager=history_manager)
    rm.add(dataset)

    np.random.seed(5)
    s_seq, a_seq, r_seq, ss_seq, ab_seq, last_seq, ps_seq, nps_seq, lengths = rm.get(n_samples)

    np.random.seed(5)
    expected_idxs = np.random.randint(0, rm.size, (n_samples,))

    assert s_seq.shape == (n_samples, truncation_length, history_length, *obs_shape)
    assert ps_seq.shape == (n_samples, truncation_length, *policy_state_shape)

    for k, idx in enumerate(expected_idxs):
        idx = int(idx)
        begin = max(idx - truncation_length + 1, 0)
        length = idx + 1 - begin
        assert lengths[k] == length
        assert np.allclose(s_seq[k, :length, history_length - 1], rm._dataset.state[begin:idx + 1])
        for t in range(length):
            anchor = begin + t
            if anchor - 1 >= 0:
                assert np.allclose(s_seq[k, t, 0], rm._dataset.state[anchor - 1])
            else:
                assert np.allclose(s_seq[k, t, 0], 0.0)


def test_sequence_replay_memory_action_history_return_extra():
    obs_shape = (4,)
    act_shape = (2,)
    policy_state_shape = (8,)
    truncation_length = 4
    action_history_length = 2
    n = 15
    n_samples = 3

    rng = np.random.RandomState(2)
    dataset, *_ = make_dataset(n, rng, obs_shape, act_shape,
                               policy_state_shape=policy_state_shape, episode_ends=[14])

    mdp_info = make_mdp_info(obs_shape, act_shape)
    agent_info = make_agent_info(policy_state_shape=policy_state_shape)
    history_manager = HistoryManager.from_infos(mdp_info, agent_info, action_history_length=action_history_length)
    rm = SequenceReplayMemory(mdp_info, agent_info, initial_size=5, max_size=100,
                              truncation_length=truncation_length, history_manager=history_manager, return_extra=True)
    rm.add(dataset)

    np.random.seed(7)
    *_, lengths, extra = rm.get(n_samples)

    np.random.seed(7)
    expected_idxs = np.random.randint(0, rm.size, (n_samples,))

    prev_actions = extra['action_history']
    assert prev_actions.shape == (n_samples, truncation_length, action_history_length, *act_shape)

    for k, idx in enumerate(expected_idxs):
        idx = int(idx)
        begin = max(idx - truncation_length + 1, 0)
        length = idx + 1 - begin
        assert lengths[k] == length
        for t in range(length):
            anchor = begin + t
            for j in range(action_history_length):
                src = anchor - (action_history_length - j)
                if src >= 0:
                    assert np.allclose(prev_actions[k, t, j], rm._dataset.action[src])
                else:
                    assert np.allclose(prev_actions[k, t, j], 0.0)


def test_sequence_replay_memory_history_reduced_sampling_across_write_head():
    obs_shape = (1,)
    act_shape = (1,)
    policy_state_shape = (2,)
    history_length = 2
    truncation_length = 3
    max_size = 6

    mdp_info = make_mdp_info(obs_shape, act_shape)
    agent_info = make_agent_info(policy_state_shape=policy_state_shape)
    history_manager = HistoryManager.from_infos(mdp_info, agent_info, history_length=history_length)
    rm = SequenceReplayMemory(mdp_info, agent_info, initial_size=1, max_size=max_size,
                              truncation_length=truncation_length, history_manager=history_manager)

    # one continuous episode (no last flags) longer than the buffer, so it wraps around the write head
    for v in range(9):
        s = np.array([[float(v)]])
        rm.add(Dataset.from_array(s, np.zeros((1, 1)), np.zeros(1), s + 0.5, np.zeros(1), np.zeros(1),
                                  policy_state=np.zeros((1, *policy_state_shape)),
                                  policy_next_state=np.zeros((1, *policy_state_shape)), backend='numpy'))

    assert rm._full and rm._idx == 3          # chronological values are [3, 4, 5, 6, 7, 8], the oldest is 3

    np.random.seed(0)
    s_seq, a_seq, r_seq, ss_seq, ab_seq, last_seq, ps_seq, nps_seq, lengths = rm.get(200)

    for k, length in enumerate(lengths):
        current = s_seq[k, :length, history_length - 1, 0]
        # the sequence is a single contiguous trajectory: consecutive timesteps differ by exactly one, and the
        # temporal length is truncated at the write head instead of stitching in unrelated frames
        assert np.allclose(current[1:] - current[:-1], 1.0)
        # each timestep's frame-stack is contiguous too, so it never crosses the write head
        assert np.allclose(s_seq[k, :length, 1, 0] - s_seq[k, :length, 0, 0], 1.0)
        # the oldest sample can never be rebuilt as an anchor's current frame, so is never sampled as one
        assert not np.any(current == 3.0)


def test_history_manager_from_infos_none():
    mdp_info = make_mdp_info((4,), (2,))
    agent_info = make_agent_info()

    assert HistoryManager.from_infos(mdp_info, agent_info, history_length=1) is None
    assert HistoryManager.from_infos(mdp_info, agent_info, history_length=3) is not None
    assert HistoryManager.from_infos(mdp_info, agent_info, action_history_length=1) is not None
    assert HistoryManager.from_infos(mdp_info, agent_info, action_history_length=2) is not None


def test_replay_memory_rejects_exogenous_history_stream():
    mdp_info = make_mdp_info((4,), (2,))
    agent_info = make_agent_info()
    history_manager = HistoryManager(mdp_info, agent_info, obs_history_length=3,
                                     extra_buffers={'command': dict(length=2, shape=(1,), dtype=float)})

    with pytest.raises(AssertionError):
        ReplayMemory(mdp_info, agent_info, initial_size=5, max_size=50, history_manager=history_manager)

    obs_action_manager = HistoryManager.from_infos(mdp_info, agent_info, history_length=3, action_history_length=1)
    ReplayMemory(mdp_info, agent_info, initial_size=5, max_size=50, history_manager=obs_action_manager)


def test_prioritized_replay_memory_add_get():
    obs_shape = (4,)
    act_shape = (2,)
    n = 10

    rng = np.random.RandomState(11)
    dataset, states, actions, rewards, next_states, _, _ = make_dataset(n, rng, obs_shape, act_shape)

    beta = LinearParameter(1.0, threshold_value=0.0, n=100)
    rm = PrioritizedReplayMemory(make_mdp_info(obs_shape, act_shape), make_agent_info(),
                                 initial_size=5, max_size=50, alpha=0.6, beta=beta)
    rm.add(dataset, np.ones(n))

    assert rm.initialized
    assert np.isclose(rm._tree.total_p, 10.0)

    np.random.seed(3)
    s, a, r, ss, ab, last, idxs, is_weight = rm.get(4)

    assert np.allclose(is_weight, 1.0)

    s_test = np.array([[-0.00828463, -0.31963136, -0.53662936,  0.31540267],
                       [0.73683739,  1.57463407, -0.03107509, -0.68344663],
                       [1.0956297,  -0.30957664,  0.72575222,  1.54907163],
                       [-0.18577532, -0.38053642,  0.08897764,  0.06367166]])
    r_test = np.array([1.31094364, 2.15667443, -0.82943725, -1.08019383])

    assert np.allclose(s, s_test)
    assert np.allclose(r, r_test)


def test_prioritized_replay_memory_update_changes_priorities():
    obs_shape = (4,)
    act_shape = (2,)
    n = 10

    rng = np.random.RandomState(11)
    dataset, *_ = make_dataset(n, rng, obs_shape, act_shape)

    beta = LinearParameter(1.0, threshold_value=0.0, n=100)
    rm = PrioritizedReplayMemory(make_mdp_info(obs_shape, act_shape), make_agent_info(),
                                 initial_size=1, max_size=50, alpha=0.6, beta=beta)
    rm.add(dataset, np.ones(n))

    np.random.seed(3)
    *_, idxs, _ = rm.get(4)

    errors = np.array([0.5, 0.1, 0.3, 0.8])
    rm.update(errors, idxs)

    new_priorities = (np.abs(errors) + 0.01) ** 0.6
    expected_total = 10.0 - 4.0 + np.sum(new_priorities)
    assert np.isclose(rm._tree.total_p, expected_total)


def test_prioritized_replay_memory_max_priority_before_init():
    obs_shape = (4,)
    act_shape = (2,)

    beta = LinearParameter(1.0, threshold_value=0.0, n=100)
    rm = PrioritizedReplayMemory(make_mdp_info(obs_shape, act_shape), make_agent_info(),
                                 initial_size=10, max_size=50, alpha=0.6, beta=beta)

    assert rm.max_priority == 1.0


def test_prioritized_replay_memory_max_priority_after_add():
    obs_shape = (4,)
    act_shape = (2,)

    beta = LinearParameter(1.0, threshold_value=0.0, n=100)
    rm = PrioritizedReplayMemory(make_mdp_info(obs_shape, act_shape), make_agent_info(),
                                 initial_size=1, max_size=50, alpha=0.6, beta=beta)

    rng = np.random.RandomState(99)
    dataset, *_ = make_dataset(5, rng, obs_shape, act_shape)
    rm.add(dataset, np.array([1.0, 3.0, 0.5, 1.5, 1.0]))

    assert rm.initialized
    assert np.isclose(rm.max_priority, 3.0)


def test_replay_memory_torch_uint8_obs():
    np.random.seed(42)

    obs_shape = (4, 8)
    obs_space = Box(low=0, high=255, shape=obs_shape, data_type=np.uint8)
    act_space = Discrete(3)
    mdp_info = MDPInfo(obs_space, act_space, gamma=0.99, horizon=100)
    agent_info = AgentInfo(is_episodic=False, policy_state_shape=None, backend='torch')

    rm = ReplayMemory(mdp_info, agent_info, initial_size=5, max_size=50)

    states = np.random.randint(0, 256, (10, *obs_shape), dtype=np.uint8)
    actions = np.random.randint(0, 3, (10, 1))
    rewards = np.random.randn(10).astype(np.float32)
    next_states = np.random.randint(0, 256, (10, *obs_shape), dtype=np.uint8)
    absorbings = np.zeros(10)
    lasts = np.zeros(10)
    lasts[-1] = 1.0

    dataset = Dataset.from_array(states, actions, rewards, next_states, absorbings, lasts, backend='numpy')
    rm.add(dataset)

    assert rm.initialized
    state, action, reward, next_state, absorbing, last = rm.get(5)
    assert state.dtype == torch.uint8
    assert state.shape == (5, *obs_shape)


def test_replay_memory_history():
    obs_shape = (4,)
    act_shape = (2,)
    n = 10
    history_length = 4

    rng = np.random.RandomState(42)
    dataset, states, actions, rewards, next_states, _, _ = make_dataset(n, rng, obs_shape, act_shape)

    mdp_info = make_mdp_info(obs_shape, act_shape)
    agent_info = make_agent_info()
    history_manager = HistoryManager.from_infos(mdp_info, agent_info, history_length=history_length)
    rm = ReplayMemory(mdp_info, agent_info, initial_size=5, max_size=50, history_manager=history_manager)
    rm.add(dataset)

    np.random.seed(7)
    s, a, r, ss, ab, last = rm.get(4)

    expected_idxs = np.array([4, 9, 6, 3])

    assert s.shape == (4, history_length, *obs_shape)
    assert ss.shape == (4, history_length, *obs_shape)
    assert a.shape == (4, *act_shape)
    assert np.allclose(s[:, history_length - 1], states[expected_idxs])
    assert np.allclose(ss[:, history_length - 1], next_states[expected_idxs])


def test_replay_memory_history_reduced_sampling_excludes_write_head():
    obs_shape = (1,)
    act_shape = (1,)
    history_length = 3
    max_size = 5

    mdp_info = make_mdp_info(obs_shape, act_shape)
    agent_info = make_agent_info()
    history_manager = HistoryManager.from_infos(mdp_info, agent_info, history_length=history_length)
    rm = ReplayMemory(mdp_info, agent_info, initial_size=1, max_size=max_size, history_manager=history_manager)

    # one continuous episode (no last flags) longer than the buffer, so it wraps around the write head
    for v in range(8):
        s = np.array([[float(v)]])
        rm.add(Dataset.from_array(s, np.zeros((1, 1)), np.zeros(1), s + 0.5, np.zeros(1), np.zeros(1),
                                  backend='numpy'))

    assert rm._full and rm._idx == 3          # chronological order of positions is [3, 4, 0, 1, 2] -> values [3..7]

    np.random.seed(0)
    s, a, r, ss, ab, last = rm.get(200)

    # each stacked observation must be a contiguous chronological window: since the whole buffer is a single
    # continuous episode, consecutive stacked frames must differ by exactly one (no write-head crossing)
    assert np.allclose(s[:, 1:, 0] - s[:, :-1, 0], 1.0)

    # the (history_length - 1) oldest samples (values 3 and 4) can never be rebuilt, so must never be sampled
    anchors = s[:, history_length - 1, 0]
    assert not np.any(np.isin(anchors, [3.0, 4.0]))


def test_replay_memory_wrap_after_full():
    obs_shape = (4,)
    act_shape = (2,)
    max_size = 10

    rng = np.random.RandomState(42)
    ds_a, states_a, _, _, _, _, _ = make_dataset(10, rng, obs_shape, act_shape)
    ds_b, states_b, _, _, _, _, _ = make_dataset(6, rng, obs_shape, act_shape)
    ds_c, states_c, _, _, _, _, _ = make_dataset(6, rng, obs_shape, act_shape)

    mdp_info = make_mdp_info(obs_shape, act_shape)
    agent_info = make_agent_info()
    rm = ReplayMemory(mdp_info, agent_info, initial_size=5, max_size=max_size)
    rm.add(ds_a)
    rm.add(ds_b)
    rm.add(ds_c)

    assert rm._full
    assert rm.size == max_size
    assert rm._idx == 2

    assert np.allclose(rm._dataset.state[6:10], states_c[:4])
    assert np.allclose(rm._dataset.state[:2], states_c[4:6])


def test_replay_memory_wrap_after_full_stateful():
    obs_shape = (4,)
    act_shape = (2,)
    policy_state_shape = (3,)
    max_size = 10

    rng = np.random.RandomState(42)
    ds_a, _, _, _, _, ps_a, _ = make_dataset(10, rng, obs_shape, act_shape, policy_state_shape)
    ds_b, _, _, _, _, ps_b, _ = make_dataset(6, rng, obs_shape, act_shape, policy_state_shape)
    ds_c, _, _, _, _, ps_c, _ = make_dataset(6, rng, obs_shape, act_shape, policy_state_shape)

    mdp_info = make_mdp_info(obs_shape, act_shape)
    agent_info = make_agent_info(policy_state_shape=policy_state_shape)
    rm = ReplayMemory(mdp_info, agent_info, initial_size=5, max_size=max_size, store_policy_state=True)
    rm.add(ds_a)
    rm.add(ds_b)
    rm.add(ds_c)

    assert rm._full
    assert rm.size == max_size
    assert rm._idx == 2

    assert np.allclose(rm._dataset.policy_state[6:10], ps_c[:4])
    assert np.allclose(rm._dataset.policy_state[:2], ps_c[4:6])


def test_replay_memory_opt_out_drops_policy_state():
    obs_shape = (4,)
    act_shape = (2,)
    policy_state_shape = (3,)

    rng = np.random.RandomState(0)
    ds, *_ = make_dataset(8, rng, obs_shape, act_shape, policy_state_shape=policy_state_shape)

    mdp_info = make_mdp_info(obs_shape, act_shape)
    agent_info = make_agent_info(policy_state_shape=policy_state_shape)
    rm = ReplayMemory(mdp_info, agent_info, initial_size=5, max_size=50)

    assert not rm._dataset.is_stateful

    rm.add(ds)
    assert rm.size == 8
    assert len(rm.get(4)) == 6
