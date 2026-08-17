import torch
import numpy as np
from mushroom_rl.core import MDPInfo, AgentInfo
from mushroom_rl.core.spaces import Box
from mushroom_rl.core.history_manager import HistoryManager
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.environments.segway import Segway
from mushroom_rl.core import Core, Agent
from mushroom_rl.approximators.parametric import LinearApproximator, TorchApproximator
from mushroom_rl.approximators.parametric.networks import LinearNetwork
from mushroom_rl.rl_utils.value_functions import compute_gae, compute_advantage_montecarlo, _next_action_history


def test_next_action_history_shift_across_episode_boundaries():
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


def test_next_action_history_length_one_is_the_action():
    action_history = torch.tensor([[10.], [11.], [12.]])
    action = torch.tensor([[10.], [11.], [12.]])
    last = torch.tensor([0., 1., 1.])

    next_action_history = _next_action_history(action_history, action, last)
    assert torch.allclose(next_action_history, action)


def test_compute_advantage_montecarlo():
    def advantage_montecarlo(V, s, ss, r, absorbing, last, gamma):
        with torch.no_grad():
            r = r.squeeze()
            q = torch.zeros(len(r))
            v = V(s).squeeze()

            for rev_k in range(len(r)):
                k = len(r) - rev_k - 1
                if last[k] or rev_k == 0:
                    q_next = V(ss[k]).squeeze().item()
                q_next = r[k] + gamma * q_next * (1 - absorbing[k].int())
                q[k] = q_next

            adv = q - v
            return q[:, None], adv[:, None]

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
    V = TorchApproximator(input_shape=mdp.info.observation_space.shape, output_shape=(1,),
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
