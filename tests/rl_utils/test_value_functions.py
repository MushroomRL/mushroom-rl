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


def test_value_functions_ragged_episodes():
    torch.manual_seed(42)

    V, state, next_state, reward = _edge_case_setup(8)
    last = torch.tensor([0, 0, 1, 1, 0, 0, 0, 1], dtype=torch.bool)
    absorbing = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool)

    q, adv = compute_advantage_montecarlo(V, state, next_state, reward, absorbing, last, 0.99)
    v_target, gen_adv = compute_gae(V, state, next_state, reward, absorbing, last, 0.99, 0.95)

    assert torch.allclose(q.squeeze(), torch.tensor([0.93121833, 0.38666949, 0.83650607, 0.33974251,
                                                     6.83558655, 5.93961239, 7.10078287, 4.55114269]))
    assert torch.allclose(adv.squeeze(), torch.tensor([-0.75663167, -0.12601939, -0.02929711, -0.48005757,
                                                       6.43413496, 4.48077297, 6.69658470, 2.41804862]))
    assert torch.allclose(v_target.squeeze(), torch.tensor([1.27571929, 0.45201996, 0.83650607, 0.33974254,
                                                            2.58936834, 1.82835221, 5.00620508, 4.55114269]))
    assert torch.allclose(gen_adv.squeeze(), torch.tensor([-0.41213071, -0.06066891, -0.02929711, -0.48005754,
                                                           2.18791699, 0.36951303, 4.60200691, 2.41804838]))


def test_value_functions_ragged_episodes_with_absorbing():
    torch.manual_seed(42)

    V, state, next_state, reward = _edge_case_setup(8)
    last = torch.tensor([0, 0, 1, 1, 0, 0, 0, 1], dtype=torch.bool)
    absorbing = torch.tensor([0, 0, 1, 0, 0, 0, 0, 1], dtype=torch.bool)

    q, adv = compute_advantage_montecarlo(V, state, next_state, reward, absorbing, last, 0.99)
    v_target, gen_adv = compute_gae(V, state, next_state, reward, absorbing, last, 0.99, 0.95)

    assert torch.allclose(q.squeeze(), torch.tensor([1.66141570, 1.12424254, 1.58152938, 0.33974251,
                                                     5.08836174, 4.17473841, 5.31808186, 2.75043464]))
    assert torch.allclose(adv.squeeze(), torch.tensor([-0.02643430, 0.61155367, 0.71572620, -0.48005757,
                                                       4.68691015, 2.71589923, 4.91388369, 0.61734056]))
    assert torch.allclose(v_target.squeeze(), torch.tensor([1.93472242, 1.15271449, 1.58152938, 0.33974254,
                                                            1.09134173, 0.23555398, 3.31263971, 2.75043464]))
    assert torch.allclose(gen_adv.squeeze(), torch.tensor([0.24687243, 0.64002556, 0.71572620, -0.48005754,
                                                           0.68989027, -1.22328520, 2.90844154, 0.61734056]))


def test_value_functions_truncated_buffer():
    torch.manual_seed(42)

    V, state, next_state, reward = _edge_case_setup(8)
    last = torch.tensor([0, 0, 1, 0, 0, 1, 0, 0], dtype=torch.bool)
    absorbing = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool)

    q, adv = compute_advantage_montecarlo(V, state, next_state, reward, absorbing, last, 0.99)
    v_target, gen_adv = compute_gae(V, state, next_state, reward, absorbing, last, 0.99, 0.95)

    assert torch.allclose(q.squeeze(), torch.tensor([0.93121833, 0.38666949, 0.83650607, -1.70239782,
                                                     -1.51946592, -2.49983501, 7.10078287, 4.55114269]))
    assert torch.allclose(adv.squeeze(), torch.tensor([-0.75663167, -0.12601939, -0.02929711, -2.52219796,
                                                       -1.92091739, -3.95867419, 6.69658470, 2.41804862]))
    assert torch.allclose(v_target.squeeze(), torch.tensor([1.27571929, 0.45201996, 0.83650607, -1.43097758,
                                                            -1.48129189, -2.49983525, 5.00620508, 4.55114269]))
    assert torch.allclose(gen_adv.squeeze(), torch.tensor([-0.41213071, -0.06066891, -0.02929711, -2.25077772,
                                                           -1.88274336, -3.95867443, 4.60200691, 2.41804838]))


def test_value_functions_single_boundary_mid_buffer():
    torch.manual_seed(42)

    V, state, next_state, reward = _edge_case_setup(8)
    last = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.bool)
    absorbing = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool)

    q, adv = compute_advantage_montecarlo(V, state, next_state, reward, absorbing, last, 0.99)
    v_target, gen_adv = compute_gae(V, state, next_state, reward, absorbing, last, 0.99, 0.95)

    assert torch.allclose(q.squeeze(), torch.tensor([1.99106741, 1.45722413, 1.91787446, 0.33974251,
                                                     6.83558655, 5.93961239, 7.10078287, 4.55114269]))
    assert torch.allclose(adv.squeeze(), torch.tensor([0.30321741, 0.94453526, 1.05207133, -0.48005757,
                                                       6.43413496, 4.48077297, 6.69658470, 2.41804862]))
    assert torch.allclose(v_target.squeeze(), torch.tensor([0.87635458, 0.02738973, 0.38501194, 0.33974254,
                                                            2.58936834, 1.82835221, 5.00620508, 4.55114269]))
    assert torch.allclose(gen_adv.squeeze(), torch.tensor([-0.81149542, -0.48529914, -0.48079124, -0.48005754,
                                                           2.18791699, 0.36951303, 4.60200691, 2.41804838]))


def test_value_functions_no_boundary():
    torch.manual_seed(42)

    V, state, next_state, reward = _edge_case_setup(8)
    last = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool)
    absorbing = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool)

    q, adv = compute_advantage_montecarlo(V, state, next_state, reward, absorbing, last, 0.99)
    v_target, gen_adv = compute_gae(V, state, next_state, reward, absorbing, last, 0.99, 0.95)

    assert torch.allclose(q.squeeze(), torch.tensor([8.03541088, 7.56262159, 8.08494282, 6.56910419,
                                                     6.83558655, 5.93961239, 7.10078287, 4.55114269]))
    assert torch.allclose(adv.squeeze(), torch.tensor([6.34756088, 7.04993248, 7.21913958, 5.74930429,
                                                       6.43413496, 4.48077297, 6.69658470, 2.41804862]))
    assert torch.allclose(v_target.squeeze(), torch.tensor([2.58820605, 1.84754014, 2.32031274, 2.39747858,
                                                            2.58936834, 1.82835221, 5.00620508, 4.55114269]))
    assert torch.allclose(gen_adv.squeeze(), torch.tensor([0.90035599, 1.33485126, 1.45450950, 1.57767844,
                                                           2.18791699, 0.36951303, 4.60200691, 2.41804838]))


def test_value_functions_every_step_terminal():
    torch.manual_seed(42)

    V, state, next_state, reward = _edge_case_setup(8)
    last = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.bool)
    absorbing = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool)

    q, adv = compute_advantage_montecarlo(V, state, next_state, reward, absorbing, last, 0.99)
    v_target, gen_adv = compute_gae(V, state, next_state, reward, absorbing, last, 0.99, 0.95)

    assert torch.allclose(q.squeeze(), torch.tensor([1.33277845, 0.47957394, 0.83650607, 0.33974251,
                                                     2.24184132, -2.49983501, 2.73203087, 4.55114269]))
    assert torch.allclose(adv.squeeze(), torch.tensor([-0.35507154, -0.03311494, -0.02929711, -0.48005757,
                                                       1.84038985, -3.95867419, 2.32783270, 2.41804862]))
    assert torch.allclose(v_target.squeeze(), torch.tensor([1.33277845, 0.47957391, 0.83650607, 0.33974254,
                                                            2.24184132, -2.49983525, 2.73203087, 4.55114269]))
    assert torch.allclose(gen_adv.squeeze(), torch.tensor([-0.35507160, -0.03311497, -0.02929711, -0.48005754,
                                                           1.84038997, -3.95867443, 2.32783270, 2.41804838]))


def _edge_case_setup(n_steps):
    V = TorchApproximator(input_shape=(3,), output_shape=(1,), network=LinearNetwork, use_bias=True,
                          loss=torch.nn.MSELoss(),
                          optimizer={'class': torch.optim.Adam, 'params': {'lr': 0.001}})

    return V, torch.randn(n_steps, 3), torch.randn(n_steps, 3), torch.randn(n_steps)
