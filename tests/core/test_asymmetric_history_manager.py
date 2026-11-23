import numpy as np
import pytest
import torch

from helper.arrays import to_numpy
from helper.cases import devices
from helper.dummies import CountingEnv, CountingVecEnv

from mushroom_rl.core import Core, Agent, AgentInfo, Box, MDPInfo
from mushroom_rl.core.asymmetric_history_manager import AsymmetricHistoryManager
from mushroom_rl.core.circular_dataset import CircularDataset
from mushroom_rl.policy import Policy


class WindowZeroPolicy(Policy):
    def __init__(self, history_length):
        super().__init__('numpy')
        self._window_axes = 2 if history_length > 1 else 1

    def draw_action(self, state):
        return np.zeros(state.shape[:-self._window_axes] + (1,))


class AsymmetricRecordingAgent(Agent):
    def __init__(self, mdp_info, history_length=1, actor_indices=None, critic_indices=None, critic_history_length=1):
        super().__init__(mdp_info, WindowZeroPolicy(history_length), backend='numpy')
        self._history_manager = AsymmetricHistoryManager.default_streams(mdp_info, self._info, history_length, None,
                                                                         actor_indices, critic_indices,
                                                                         critic_history_length)
        self.online = dict()
        self.blocks = list()

    def draw_action(self, state):
        action = super().draw_action(state)
        windows = self._history_manager._last_windows
        states = state if state.ndim == 2 else state[None]
        for i, row in enumerate(states):
            self.online[row.tobytes()] = {name: np.copy(window if state.ndim == 1 else window[i])
                                          for name, window in windows.items()}
        return action

    def fit(self, dataset):
        self.blocks.append((dataset.copy(), self._history_manager.parse_history(dataset)))


def mismatching_rows(agent, actor_indices, critic_indices, critic_history_length):
    mismatches = list()
    for k, (dataset, batch) in enumerate(agent.blocks):
        for row, state in enumerate(np.asarray(dataset.state)):
            online = agent.online[state.tobytes()]
            actor = online.get('obs_history', state if actor_indices is None else state[actor_indices])
            critic = state if critic_indices is None else state[critic_indices]
            critic = online['critic_history'] if critic_history_length > 1 else critic
            if not np.array_equal(actor, batch.state[row]) or not np.array_equal(critic, batch.critic_state[row]):
                mismatches.append((k, row))
    return mismatches


def make_parse_case(backend='numpy', device=None):
    env = CountingEnv(horizon=4)
    agent = AsymmetricRecordingAgent(env.info)
    Core(agent, env).learn(n_steps=10, n_steps_per_fit=10, quiet=True)
    dataset = agent.blocks[0][0]
    agent_info = AgentInfo(is_episodic=False, policy_state_shape=None, backend=backend, device=device)
    hm = AsymmetricHistoryManager.default_streams(env.info, agent_info, history_length=2, actor_indices=[2],
                                                  critic_indices=[1, 2], critic_history_length=3)
    return env, dataset, hm


def test_online_offline():
    cases = ((CountingEnv(horizon=10), 40, 8, dict(history_length=3, actor_indices=[2, 1], critic_indices=[1],
                                                   critic_history_length=2)),
             (CountingEnv(horizon=10), 40, 8, dict(history_length=2, actor_indices=[1, 2])),
             (CountingVecEnv(3, horizon=10), 81, 27, dict(history_length=3, actor_indices=[2, 1], critic_indices=[1],
                                                          critic_history_length=4)),
             (CountingVecEnv(2, horizon=7), 40, 10, dict(actor_indices=[0, 2], critic_indices=[2, 1, 0],
                                                         critic_history_length=3)))
    for env, n_steps, n_steps_per_fit, config in cases:
        agent = AsymmetricRecordingAgent(env.info, **config)
        Core(agent, env).learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, quiet=True)

        assert len(agent.blocks) == n_steps // n_steps_per_fit
        assert mismatching_rows(agent, config.get('actor_indices'), config.get('critic_indices'),
                                config.get('critic_history_length', 1)) == []

    env = CountingEnv(horizon=10)
    agent = AsymmetricRecordingAgent(env.info, history_length=3, actor_indices=[2, 1], critic_indices=[1],
                                     critic_history_length=2)
    Core(agent, env).learn(n_steps=16, n_steps_per_fit=8, quiet=True)
    batch = agent.blocks[1][1]

    assert np.array_equal(batch.state[0], np.array([[6., 6.], [7., 7.], [8., 8.]]))
    assert np.array_equal(batch.critic_state[0], np.array([[7.], [8.]]))
    assert np.array_equal(batch.critic_next_state[0], np.array([[8.], [9.]]))
    assert np.array_equal(batch.critic_state[1], batch.critic_next_state[0])
    states, last = np.asarray(agent.blocks[1][0].state), agent.blocks[1][0].last
    assert np.array_equal(agent.history_manager.build_history('critic_history', states, last,
                                                              attachment=agent.blocks[1][0].history_state),
                          batch.critic_state)


def test_parse():
    env, dataset, hm = make_parse_case()
    full = hm.parse_history(dataset)
    _, anchor, endpoint = hm.build_nstep_return(dataset.reward, dataset.absorbing, dataset.last, gamma=.5,
                                                n_steps_return=2)
    nstep = hm.parse_nstep_history(dataset, gamma=.5, n_steps_return=2)
    buffer = CircularDataset(dataset._dataset_info, len(dataset))
    buffer.append_replay_batch(dataset)
    circular = hm.parse_history(buffer)
    circular_nstep = hm.parse_nstep_history(buffer, gamma=.5, n_steps_return=2)

    assert hm.actor_indices == [2] and hm.critic_indices == [1, 2] and hm.critic_history_length == 3
    assert full.state.shape == (10, 2, 1) and full.critic_state.shape == (10, 3, 2)
    assert np.array_equal(full.state[5], np.array([[0.], [1.]]))
    assert np.array_equal(full.critic_state[5], np.array([[0., 0.], [5., 0.], [6., 1.]]))
    assert np.array_equal(full.critic_next_state[5], np.array([[5., 0.], [6., 1.], [7., 2.]]))
    assert np.array_equal(hm.parse_state(dataset), full.state)
    assert np.array_equal(nstep.state, full.state[anchor])
    assert np.array_equal(nstep.critic_state, full.critic_state[anchor])
    assert np.array_equal(nstep.critic_next_state, full.critic_next_state[endpoint])
    for field in ('state', 'next_state', 'critic_state', 'critic_next_state'):
        assert np.array_equal(getattr(circular, field), getattr(full, field))
        assert np.array_equal(getattr(circular_nstep, field), getattr(nstep, field))
    assert np.array_equal(hm.parse_initial_critic_state(dataset),
                          np.array([[[0., 0.], [0., 0.], [t, 0.]] for t in (0., 5., 10.)]))
    assert np.array_equal(hm.parse_history(dataset, anchor_idxs=np.array([1, 5])).critic_state,
                          full.critic_state[[1, 5]])
    states = np.asarray(dataset.state)
    assert np.array_equal(hm.build_history('critic_history', states, dataset.last), full.critic_state)
    assert np.array_equal(hm.build_history('critic_history', states, dataset.last, anchor_idxs=np.array([1, 5])),
                          full.critic_state[[1, 5]])

    single = AsymmetricHistoryManager.default_streams(env.info, AgentInfo(False, None, 'numpy'), actor_indices=[2],
                                                      critic_indices=[1, 2])
    single_full = single.parse_history(dataset)
    single_nstep = single.parse_nstep_history(dataset, gamma=.5, n_steps_return=2)
    assert single.critic_history_length == 1 and single_full.critic_state.shape == (10, 2)
    assert np.array_equal(single_full.critic_state, full.critic_state[:, -1])
    assert np.array_equal(single_nstep.critic_next_state, single_full.critic_next_state[endpoint])
    assert np.array_equal(single.parse_initial_critic_state(dataset), np.array([[0., 0.], [5., 0.], [10., 0.]]))

    critic_all = AsymmetricHistoryManager.default_streams(env.info, AgentInfo(False, None, 'numpy'),
                                                          critic_history_length=3)
    parsed = critic_all.parse_history(dataset)
    assert critic_all.actor_indices is None and critic_all.critic_indices is None
    assert parsed.state is not parsed.critic_state
    assert np.array_equal(parsed.critic_state[..., 1:], full.critic_state)
    assert np.array_equal(parsed.state, np.asarray(dataset.state))

    symmetric = AsymmetricHistoryManager.default_streams(env.info, AgentInfo(False, None, 'numpy'))
    for parsed in (symmetric.parse_history(dataset), symmetric.parse_history(dataset, anchor_idxs=np.array([1, 5])),
                   symmetric.parse_history(dataset, to='torch'),
                   symmetric.parse_nstep_history(dataset, gamma=.5, n_steps_return=2)):
        assert parsed.critic_state is parsed.state and parsed.critic_next_state is parsed.next_state


@pytest.mark.parametrize('device', devices())
def test_indices(device):
    env = CountingEnv(horizon=4)
    agent_info = AgentInfo(False, None, 'numpy')
    for indices in ([], [True, False, True], [0.5], [[0, 1]], [-1], [3], [1, 1]):
        with pytest.raises(AssertionError):
            AsymmetricHistoryManager.default_streams(env.info, agent_info, actor_indices=indices)
        with pytest.raises(AssertionError):
            AsymmetricHistoryManager.default_streams(env.info, agent_info, critic_indices=indices)
    with pytest.raises(AssertionError):
        AsymmetricHistoryManager(env.info, agent_info, streams={'obs_history': dict(length=2, shape=(3,), dtype=float)},
                                 actor_indices=[0])
    image = MDPInfo(Box(0., 1., shape=(2, 2)), env.info.action_space, 0.99, 4)
    with pytest.raises(AssertionError):
        AsymmetricHistoryManager.default_streams(image, agent_info, critic_indices=[0])

    for indices in (np.array([2, 0]), torch.tensor([2, 0], device=device), (2, 0)):
        hm = AsymmetricHistoryManager.default_streams(env.info, agent_info, actor_indices=indices,
                                                      critic_indices=indices)
        assert hm.actor_indices == [2, 0] and hm.critic_indices == [2, 0]
        assert all(type(index) is int for index in hm.actor_indices)


@pytest.mark.parametrize('device', devices())
def test_backend(device, tmpdir):
    _, dataset, hm = make_parse_case()
    _, _, torch_hm = make_parse_case('torch', device)
    expected = hm.parse_history(dataset)
    parsed = torch_hm.parse_history(dataset)

    assert parsed.critic_state.device.type == device
    for field in ('state', 'next_state', 'critic_state', 'critic_next_state'):
        assert np.array_equal(to_numpy(getattr(parsed, field)), getattr(expected, field))

    torch_hm.save(tmpdir / 'manager.msh')
    loaded = AsymmetricHistoryManager.load(tmpdir / 'manager.msh', device=device)
    reloaded = loaded.parse_history(dataset)

    assert type(loaded) is AsymmetricHistoryManager and loaded.critic_indices == [1, 2]
    for field in ('state', 'critic_state', 'critic_next_state'):
        assert torch.equal(getattr(reloaded, field), getattr(parsed, field))
