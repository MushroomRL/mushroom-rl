import torch
import numpy as np

from mushroom_rl.core import Core, Agent
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.environments import Segway

from mushroom_rl.utils.episodes import split_episodes, unsplit_episodes


def get_episodes(mdp, n_episodes=100, to='numpy'):
    mu = np.array([6.31154476, 3.32346271, 0.49648221])

    approximator = LinearApproximator(input_shape=mdp.info.observation_space.shape,
                                      output_shape=mdp.info.action_space.shape,
                                      weights=mu)

    policy = DeterministicPolicy(approximator)

    agent = Agent(mdp.info, policy)
    core = Core(agent, mdp)
    dataset = core.evaluate(n_episodes=n_episodes)

    return dataset.parse(to=to)


def test_torch_split():
    torch.manual_seed(42)
    mdp = Segway()
    state, action, reward, next_state, absorbing, last = get_episodes(mdp, to='torch')

    last_flags, *ep_arrays = split_episodes(last, state, action, reward, next_state, absorbing, last)
    un_state, un_action, un_reward, un_next_state, un_absorbing, un_last = unsplit_episodes(last, *ep_arrays)

    expected_flags = last.clone()
    expected_flags[-1] = True

    assert (last_flags.sum(-1) == 1).all()
    assert torch.equal(unsplit_episodes(last, last_flags), expected_flags)
    assert torch.allclose(state, un_state)
    assert torch.allclose(action, un_action)
    assert torch.allclose(reward, un_reward)
    assert torch.allclose(next_state, un_next_state)
    assert torch.allclose(absorbing, un_absorbing)
    assert torch.allclose(last, un_last)


def test_numpy_split():
    torch.manual_seed(42)
    np.random.seed(42)

    mdp = Segway()
    state, action, reward, next_state, absorbing, last = get_episodes(mdp)

    last_flags, *ep_arrays = split_episodes(last, state, action, reward, next_state, absorbing, last)
    un_state, un_action, un_reward, un_next_state, un_absorbing, un_last = unsplit_episodes(last, *ep_arrays)

    expected_flags = last.copy()
    expected_flags[-1] = True

    assert (last_flags.sum(-1) == 1).all()
    assert np.array_equal(unsplit_episodes(last, last_flags), expected_flags)
    assert np.allclose(state, un_state)
    assert np.allclose(action, un_action)
    assert np.allclose(reward, un_reward)
    assert np.allclose(next_state, un_next_state)
    assert np.allclose(absorbing, un_absorbing)
    assert np.allclose(last, un_last)


def test_torch_split_truncated_episode():
    last = torch.tensor([False, False, True, False, False])
    reward = torch.arange(1., 6.)

    last_flags, reward_ep = split_episodes(last, reward)

    expected_flags = last.clone()
    expected_flags[-1] = True

    assert last_flags.shape == (2, 3)
    assert torch.equal(last_flags.sum(-1), torch.tensor([1, 1]))
    assert torch.equal(unsplit_episodes(last, last_flags), expected_flags)
    assert torch.equal(unsplit_episodes(last, reward_ep), reward)


def test_numpy_split_truncated_episode():
    last = np.array([False, False, True, False, False])
    reward = np.arange(1., 6.)

    last_flags, reward_ep = split_episodes(last, reward)

    expected_flags = last.copy()
    expected_flags[-1] = True

    assert last_flags.shape == (2, 3)
    assert np.array_equal(last_flags.sum(-1), np.array([1, 1]))
    assert np.array_equal(unsplit_episodes(last, last_flags), expected_flags)
    assert np.array_equal(unsplit_episodes(last, reward_ep), reward)
