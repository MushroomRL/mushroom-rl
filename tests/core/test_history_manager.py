import numpy as np

from mushroom_rl.core import MDPInfo, AgentInfo
from mushroom_rl.core.spaces import Box
from mushroom_rl.core.history_manager import HistoryManager


def _make_infos(obs_shape=(2,), act_shape=(1,)):
    obs_space = Box(np.full(obs_shape, -1.0), np.full(obs_shape, 1.0), obs_shape)
    act_space = Box(np.full(act_shape, -1.0), np.full(act_shape, 1.0), act_shape)
    mdp_info = MDPInfo(obs_space, act_space, gamma=0.99, horizon=100, backend='numpy')
    agent_info = AgentInfo(is_episodic=False, policy_state_shape=None, backend='numpy')
    return mdp_info, agent_info


def _make_manager(history_length=3, obs_shape=(2,)):
    mdp_info, agent_info = _make_infos(obs_shape=obs_shape)
    return HistoryManager(mdp_info, agent_info, obs_history_length=history_length)


def test_history_length_property():
    hm = _make_manager(history_length=4)
    assert hm.history_length == 4


def test_single_assembly_and_shift():
    hm = _make_manager(history_length=3, obs_shape=(2,))
    hm.reset()

    stacked_1, _ = hm(np.array([1.0, 1.0]))
    assert stacked_1.shape == (3, 2)
    assert np.allclose(stacked_1, np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]))

    stacked_2, _ = hm(np.array([2.0, 2.0]))
    assert np.allclose(stacked_2, np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))


def test_single_reset_clears_buffer():
    hm = _make_manager(history_length=3, obs_shape=(2,))
    hm.reset()
    hm(np.array([1.0, 1.0]))
    hm(np.array([2.0, 2.0]))

    hm.reset()
    stacked, _ = hm(np.array([5.0, 5.0]))
    assert np.allclose(stacked, np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0]]))


def test_vectorized_assembly():
    hm = _make_manager(history_length=3, obs_shape=(2,))
    hm.reset_vectorized(np.array([True, True]))

    stacked, _ = hm(np.array([[1.0, 1.0], [2.0, 2.0]]))
    assert stacked.shape == (2, 3, 2)
    assert np.allclose(stacked[:, -1], np.array([[1.0, 1.0], [2.0, 2.0]]))
    assert np.allclose(stacked[:, :-1], 0.0)


def test_vectorized_masked_reset():
    hm = _make_manager(history_length=3, obs_shape=(2,))
    hm.reset_vectorized(np.array([True, True]))
    hm(np.array([[1.0, 1.0], [2.0, 2.0]]))
    hm(np.array([[3.0, 3.0], [4.0, 4.0]]))

    hm.reset_vectorized(np.array([True, False]))

    stacked, _ = hm(np.array([[7.0, 7.0], [8.0, 8.0]]))
    assert np.allclose(stacked[0], np.array([[0.0, 0.0], [0.0, 0.0], [7.0, 7.0]]))
    assert np.allclose(stacked[1], np.array([[2.0, 2.0], [4.0, 4.0], [8.0, 8.0]]))


def test_vectorized_reset_does_not_corrupt_previously_returned_stack():
    hm = _make_manager(history_length=3, obs_shape=(2,))
    hm.reset_vectorized(np.array([True, True]))
    hm(np.array([[1.0, 1.0], [2.0, 2.0]]))

    stacked_1, _ = hm(np.array([[3.0, 3.0], [4.0, 4.0]]))
    expected_stacked_1 = stacked_1.copy()

    hm.reset_vectorized(np.array([True, False]))

    assert np.allclose(stacked_1, expected_stacked_1)


def test_single_call_does_not_corrupt_previously_returned_stack():
    hm = _make_manager(history_length=3, obs_shape=(2,))
    hm.reset()

    stacked_1, _ = hm(np.array([1.0, 1.0]))
    expected_stacked_1 = stacked_1.copy()

    hm(np.array([2.0, 2.0]))

    assert np.allclose(stacked_1, expected_stacked_1)


def test_online_offline_equivalence():
    history_length = 3
    obs_shape = (2,)
    states = np.arange(10, dtype=float).reshape(5, 2)

    hm = _make_manager(history_length=history_length, obs_shape=obs_shape)
    hm.reset()
    online = np.stack([hm(s)[0] for s in states])

    last = np.zeros(5)
    last[-1] = 1.0
    anchors = np.arange(5)
    offline = hm.build_history_circular_buffer('obs_history', states, last, anchors, size=5, full=False, max_size=50)

    assert offline.shape == (5, history_length, *obs_shape)
    assert np.allclose(online, offline)


def test_online_offline_equivalence_across_episodes():
    history_length = 3
    states = np.arange(5, dtype=float).reshape(5, 1)
    last = np.zeros(5)
    last[2] = 1.0                       # index 2 is the last transition of the first episode

    hm = _make_manager(history_length=history_length, obs_shape=(1,))

    online = []
    hm.reset()
    for t in range(5):
        if t > 0 and last[t - 1]:       # a new episode starts: the agent resets the manager
            hm.reset()
        online.append(hm(states[t])[0])
    online = np.stack(online)

    offline = hm.build_history_circular_buffer('obs_history', states, last, np.arange(5),
                                               size=5, full=False, max_size=50)

    assert np.allclose(online, offline)
    # the second episode (starting at index 3) is zero-padded, never stitched to the first episode
    assert np.allclose(offline[3], np.array([[0.0], [0.0], [3.0]]))
    assert np.allclose(offline[4], np.array([[0.0], [3.0], [4.0]]))


def test_build_history_batch_stops_at_episode_boundary():
    history_length = 3
    states = np.arange(10, dtype=float).reshape(5, 2)
    last = np.zeros(5)
    last[2] = 1.0

    hm = _make_manager(history_length=history_length, obs_shape=(2,))
    offline = hm.build_history_circular_buffer('obs_history', states, last, np.array([4]),
                                               size=5, full=False, max_size=50)

    assert np.allclose(offline[0], np.array([[0.0, 0.0], states[3], states[4]]))


def test_action_only_stream():
    mdp_info, agent_info = _make_infos()
    hm = HistoryManager(mdp_info, agent_info,
                        extra_buffers={'action_history': dict(length=2, shape=(2,), dtype=float)})
    hm.reset()

    state_0, kwargs_0 = hm()
    assert state_0 is None
    assert kwargs_0['action_history'].shape == (2, 2)
    assert np.allclose(kwargs_0['action_history'], 0.0)

    hm.record_action(np.array([1.0, 1.0]))
    _, kwargs_1 = hm()
    assert np.allclose(kwargs_1['action_history'], np.array([[0.0, 0.0], [1.0, 1.0]]))

    hm.record_action(np.array([2.0, 2.0]))
    _, kwargs_2 = hm()
    assert np.allclose(kwargs_2['action_history'], np.array([[1.0, 1.0], [2.0, 2.0]]))


def test_extra_buffer_stream():
    mdp_info, agent_info = _make_infos(obs_shape=(2,))
    hm = HistoryManager(mdp_info, agent_info, obs_history_length=2,
                        extra_buffers={'command': dict(length=3, shape=(1,), dtype=float)})
    hm.reset()

    state, kwargs = hm(np.array([1.0, 1.0]), command=np.array([7.0]))
    assert state.shape == (2, 2)
    assert set(kwargs.keys()) == {'command'}
    assert kwargs['command'].shape == (3, 1)
    assert np.allclose(kwargs['command'], np.array([[0.0], [0.0], [7.0]]))

    _, kwargs = hm(np.array([2.0, 2.0]), command=np.array([8.0]))
    assert np.allclose(kwargs['command'], np.array([[0.0], [7.0], [8.0]]))

    _, kwargs = hm(np.array([3.0, 3.0]))
    assert np.allclose(kwargs['command'], np.array([[7.0], [8.0], [0.0]]))


def test_extra_only_manager():
    mdp_info, agent_info = _make_infos()
    hm = HistoryManager(mdp_info, agent_info, extra_buffers={'foo': dict(length=2, shape=(1,), dtype=float)})
    hm.reset()

    state, kwargs = hm(foo=np.array([5.0]))
    assert state is None
    assert np.allclose(kwargs['foo'], np.array([[0.0], [5.0]]))


def test_uses_action_reflects_action_history_stream():
    mdp_info, agent_info = _make_infos()
    hm_action = HistoryManager(mdp_info, agent_info,
                               extra_buffers={'action_history': dict(length=2, shape=(1,), dtype=float)})
    assert hm_action.uses_action

    hm_other = HistoryManager(mdp_info, agent_info,
                              extra_buffers={'command': dict(length=2, shape=(1,), dtype=float)})
    assert not hm_other.uses_action


def test_multi_stream_splits_state_and_action():
    mdp_info, agent_info = _make_infos(obs_shape=(2,))
    hm = HistoryManager(mdp_info, agent_info, obs_history_length=3,
                        extra_buffers={'action_history': dict(length=2, shape=(1,), dtype=float)})
    hm.reset()

    hm.record_action(np.array([5.0]))
    state, kwargs = hm(np.array([1.0, 1.0]))

    assert set(kwargs.keys()) == {'action_history'}
    assert state.shape == (3, 2)
    assert kwargs['action_history'].shape == (2, 1)
    assert np.allclose(state[-1], np.array([1.0, 1.0]))
    assert np.allclose(kwargs['action_history'][-1], np.array([5.0]))


def _make_action_manager(action_history_length):
    mdp_info, agent_info = _make_infos(act_shape=(1,))
    return HistoryManager(mdp_info, agent_info,
                          extra_buffers={'action_history': dict(length=action_history_length, shape=(1,),
                                                                dtype=float, offset=1)})


def test_previous_action_online_offline_equivalence():
    actions = np.arange(5, dtype=float).reshape(5, 1)
    last = np.zeros(5)
    last[-1] = 1.0

    hm = _make_action_manager(action_history_length=2)
    hm.reset()
    online = []
    for t in range(5):
        _, kwargs = hm()
        online.append(kwargs['action_history'])
        hm.record_action(actions[t])
    online = np.stack(online)

    offline = hm.build_history('action_history', actions, last, np.arange(5))

    assert online.shape == (5, 2, 1)
    assert np.allclose(online, offline)


def test_previous_action_zeroed_at_trajectory_start():
    actions = np.arange(5, dtype=float).reshape(5, 1)
    last = np.zeros(5)
    last[1] = 1.0                       # first episode ends at index 1, second starts at index 2

    hm = _make_action_manager(action_history_length=2)
    offline = hm.build_history('action_history', actions, last, np.arange(5))

    # index 2 starts a new trajectory: its previous action is zeroed, not stitched to the first episode
    assert np.allclose(offline[2], np.array([[0.0], [0.0]]))
    assert np.allclose(offline[3], np.array([[0.0], [2.0]]))


def test_previous_action_online_offline_equivalence_across_episodes():
    actions = np.arange(5, dtype=float).reshape(5, 1)
    last = np.zeros(5)
    last[2] = 1.0                       # first episode ends at index 2, second starts at index 3

    hm = _make_action_manager(action_history_length=2)

    online = []
    hm.reset()
    for t in range(5):
        if t > 0 and last[t - 1]:       # a new episode starts: the agent resets the manager
            hm.reset()
        _, kwargs = hm()
        online.append(kwargs['action_history'])
        hm.record_action(actions[t])
    online = np.stack(online)

    offline = hm.build_history('action_history', actions, last, np.arange(5))

    assert online.shape == (5, 2, 1)
    assert np.allclose(online, offline)
    # the second episode does not stitch the first episode's actions into its previous-action window
    assert np.allclose(online[3], np.array([[0.0], [0.0]]))
    assert np.allclose(online[4], np.array([[0.0], [3.0]]))


def test_build_history_all_anchors_matches_indexed_obs():
    history_length = 3
    states = np.arange(6, dtype=float).reshape(6, 1)
    last = np.zeros(6)
    last[2] = 1.0

    hm = _make_manager(history_length=history_length, obs_shape=(1,))
    indexed = hm.build_history('obs_history', states, last, np.arange(6))
    all_anchors = hm.build_history('obs_history', states, last)

    assert all_anchors.shape == (6, history_length, 1)
    assert np.allclose(indexed, all_anchors)


def test_build_history_all_anchors_matches_indexed_action():
    actions = np.arange(6, dtype=float).reshape(6, 1)
    last = np.zeros(6)
    last[2] = 1.0

    hm = _make_action_manager(action_history_length=2)
    indexed = hm.build_history('action_history', actions, last, np.arange(6))
    all_anchors = hm.build_history('action_history', actions, last)

    assert np.allclose(indexed, all_anchors)


def test_action_history_length_one_squeezed():
    actions = np.arange(4, dtype=float).reshape(4, 1)
    last = np.zeros(4)
    last[-1] = 1.0

    hm = _make_action_manager(action_history_length=1)
    hm.reset()
    online = []
    for t in range(4):
        _, kwargs = hm()
        online.append(kwargs['action_history'])
        hm.record_action(actions[t])
    online = np.stack(online)

    offline = hm.build_history('action_history', actions, last, np.arange(4))

    assert online.shape == (4, 1)       # length-1 stream is squeezed: no stacking axis
    assert offline.shape == (4, 1)
    assert np.allclose(online, offline)
    assert np.allclose(online, np.array([[0.0], [0.0], [1.0], [2.0]]))
