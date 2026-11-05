import numpy as np
import pytest
import torch

from helper.arrays import assert_equal, make_array, to_numpy
from helper.cases import backend_devices

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


@pytest.mark.parametrize('backend, device', backend_devices('numpy', 'torch'))
def test_split(backend, device):
    torch.manual_seed(42)
    np.random.seed(42)
    arrays = get_episodes(Segway(), to=backend)
    if backend == 'torch':
        arrays = [array.to(device) for array in arrays]
    last = arrays[-1]

    last_flags, *ep_arrays = split_episodes(last, *arrays)
    unsplit = unsplit_episodes(last, *ep_arrays)

    expected_flags = to_numpy(last).copy()
    expected_flags[-1] = True

    assert (to_numpy(last_flags).sum(-1) == 1).all()
    assert_equal(unsplit_episodes(last, last_flags), expected_flags, backend, device)
    for array, restored in zip(arrays, unsplit):
        assert_equal(restored, array, backend, device)

    last = make_array(backend, [False, False, True, False, False], device)
    reward = make_array(backend, [1., 2., 3., 4., 5.], device)

    last_flags, reward_ep = split_episodes(last, reward)

    assert tuple(last_flags.shape) == (2, 3)
    assert_equal(last_flags.sum(-1), [1, 1], backend, device)
    assert_equal(unsplit_episodes(last, last_flags), [False, False, True, False, True], backend, device)
    assert_equal(unsplit_episodes(last, reward_ep), reward, backend, device)
