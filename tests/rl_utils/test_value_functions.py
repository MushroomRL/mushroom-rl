import numpy as np
import pytest
import torch

from mushroom_rl.core import MDPInfo, AgentInfo
from mushroom_rl.core.spaces import Box
from mushroom_rl.core.history_manager import HistoryManager
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.environments.segway import Segway
from mushroom_rl.core import Core, Agent
from mushroom_rl.approximators.parametric import LinearApproximator, TorchApproximator
from mushroom_rl.approximators.parametric.networks import LinearNetwork
from mushroom_rl.rl_utils.value_functions import compute_advantage, compute_gae, compute_advantage_montecarlo, \
    _next_action_history


def test_next_action_history():
    actions = torch.tensor([[10.], [11.], [12.], [13.], [14.]])
    last = torch.tensor([0., 0., 1., 0., 1.])

    act_space = Box(np.full((1,), -1.0), np.full((1,), 1.0), (1,))
    obs_space = Box(np.full((1,), -1.0), np.full((1,), 1.0), (1,))
    mdp_info = MDPInfo(obs_space, act_space, gamma=0.99, horizon=100, backend='torch')
    agent_info = AgentInfo(is_episodic=False, policy_state_shape=None, backend='torch')
    hm = HistoryManager.default_streams(mdp_info, agent_info, action_history_length=2)

    action_history = hm.build_history('action_history', actions, last, torch.arange(5))
    next_action_history = _next_action_history(action_history, actions, last)

    # non-boundary rows reuse the next row of action_history verbatim; the row ending each episode (2 and 4,
    # the latter also the last row of the buffer) is rebuilt by shifting the window and appending the action
    # just taken, rather than reading into the next (unrelated) episode
    expected = torch.tensor([[[0.], [10.]], [[10.], [11.]], [[11.], [12.]], [[0.], [13.]], [[13.], [14.]]])
    assert next_action_history.shape == (5, 2, 1)
    assert torch.allclose(next_action_history, expected)


def test_advantage_montecarlo():
    def advantage_montecarlo(V, s, ss, r, absorbing, last, gamma):
        with torch.no_grad():
            q = torch.zeros(len(r))
            v = V(s)

            for rev_k in range(len(r)):
                k = len(r) - rev_k - 1
                if last[k] or rev_k == 0:
                    q_next = V(ss[k]).item()
                q_next = r[k] + gamma * q_next * (1 - absorbing[k].int())
                q[k] = q_next

            adv = q - v
            return q, adv

    torch.manual_seed(42)
    _value_functions_tester(compute_advantage_montecarlo, advantage_montecarlo, 0.99)


def test_compute_gae():
    def gae(V, s, ss, r, absorbing, last, gamma, lam):
        with torch.no_grad():
            v = V(s)
            v_next = V(ss)
            gen_adv = torch.empty_like(v)
            for rev_k in range(len(v)):
                k = len(v) - rev_k - 1
                if last[k] or rev_k == 0:
                    gen_adv[k] = r[k] - v[k]
                    if not absorbing[k]:
                        gen_adv[k] += gamma * v_next[k]
                else:
                    gen_adv[k] = r[k] - v[k] + gamma * v_next[k] + gamma * lam * gen_adv[k + 1]
            return gen_adv + v, gen_adv

    torch.manual_seed(42)
    _value_functions_tester(compute_gae, gae, 0.99, 0.95)


def _value_functions_tester(test_fun, correct_fun, *args):
    mdp = Segway()
    V = TorchApproximator(input_shape=mdp.info.observation_space.shape, output_shape=(),
                          network=LinearNetwork, use_bias=True, loss=torch.nn.MSELoss(),
                          optimizer={'class': torch.optim.Adam, 'params': {'lr': 0.001}})

    state, action, reward, next_state, absorbing, last = _get_episodes(mdp, 10)

    correct_v, correct_adv = correct_fun(V, state, next_state, reward, absorbing, last, *args)
    v, adv = test_fun(V, state, next_state, reward, absorbing, last, *args)

    assert torch.allclose(v, correct_v)
    assert torch.allclose(adv, correct_adv)

    V.fit(state, correct_v)

    correct_v, correct_adv = correct_fun(V, state, next_state, reward, absorbing, last, *args)
    v, adv = test_fun(V, state, next_state, reward, absorbing, last, *args)

    assert torch.allclose(v, correct_v)
    assert torch.allclose(adv, correct_adv)


def _get_episodes(mdp, n_episodes=100):
    mu = np.array([6.31154476, 3.32346271, 0.49648221])

    approximator = LinearApproximator(input_shape=mdp.info.observation_space.shape,
                                      output_shape=mdp.info.action_space.shape,
                                      weights=mu)

    policy = DeterministicPolicy(approximator)

    agent = Agent(mdp.info, policy)
    core = Core(agent, mdp)
    dataset = core.evaluate(n_episodes=n_episodes)

    return dataset.parse(to='torch')


def _edge_case_setup(last):
    V = TorchApproximator(input_shape=(3,), output_shape=(), network=LinearNetwork, use_bias=True,
                          loss=torch.nn.MSELoss(),
                          optimizer={'class': torch.optim.Adam, 'params': {'lr': 0.001}})

    n_steps = len(last)
    state = torch.randn(n_steps, 3)
    next_state = torch.cat([state[1:], torch.randn(1, 3)])
    ends = last.clone()
    ends[-1] = True
    next_state[ends] = torch.randn(int(ends.sum()), 3)

    return V, state, next_state, torch.randn(n_steps)


def edge_cases():
    return [pytest.param('ragged', [0, 0, 1, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0], id='ragged'),
            pytest.param('ragged_absorbing', [0, 0, 1, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 1], id='ragged_absorbing'),
            pytest.param('truncated', [0, 0, 1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], id='truncated'),
            pytest.param('single_boundary', [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], id='single_boundary'),
            pytest.param('no_boundary', [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], id='no_boundary'),
            pytest.param('all_terminal', [1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0], id='all_terminal')]


@pytest.mark.parametrize('case, last, absorbing', edge_cases())
def test_edge_cases(case, last, absorbing):
    torch.manual_seed(42)

    last = torch.tensor(last, dtype=torch.bool)
    absorbing = torch.tensor(absorbing, dtype=torch.bool)
    V, state, next_state, reward = _edge_case_setup(last)

    q, adv = compute_advantage_montecarlo(V, state, next_state, reward, absorbing, last, 0.99)
    v_target, gen_adv = compute_gae(V, state, next_state, reward, absorbing, last, 0.99, 0.95)

    expected = torch.tensor(np.load(f'tests/rl_utils/data/test_edge_cases_{case}.npy'))
    q_test, adv_test, v_target_test, gen_adv_test = expected

    assert torch.allclose(q, q_test)
    assert torch.allclose(adv, adv_test)
    assert torch.allclose(v_target, v_target_test)
    assert torch.allclose(gen_adv, gen_adv_test)


def test_advantage():
    torch.manual_seed(42)
    absorbing = torch.tensor([0, 0, 1, 0, 1], dtype=torch.bool)
    V, state, next_state, reward = _edge_case_setup(absorbing)

    q, adv = compute_advantage(V, state, next_state, reward, absorbing, absorbing.clone(), .9)

    with torch.no_grad():
        q_test = reward + .9 * V(next_state) * ~absorbing
        adv_test = q_test - V(state)

    assert q.shape == (5,) and adv.shape == (5,)
    assert torch.allclose(q, q_test) and torch.allclose(adv, adv_test)

    V = TorchApproximator(input_shape=(3,), output_shape=(1,), network=LinearNetwork)
    for compute, params in [(compute_advantage, ()), (compute_gae, (.95,)), (compute_advantage_montecarlo, ())]:
        with pytest.raises(AssertionError):
            compute(V, state, next_state, reward, absorbing, absorbing.clone(), .9, *params)
