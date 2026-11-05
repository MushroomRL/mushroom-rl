import numpy as np
import pytest

from helper.envs import discrete_action, rollout

pytest.importorskip('minigrid')

from mushroom_rl.environments.minigrid_env import MiniGrid, MiniGridRGB  # noqa: E402


@pytest.mark.parametrize('env, path', [(MiniGrid, 'tests/environments/data/test_minigrid.npy'),
                                       (MiniGridRGB, 'tests/environments/data/test_minigrid_rgb.npy')])
def test_minigrid(env, path):
    np.random.seed(1)
    mdp = env('MiniGrid-Empty-5x5-v0', fixed_seed=1)

    assert mdp.info.observation_space.data_type == np.uint8
    assert np.allclose(rollout(mdp, discrete_action), np.load(path))


def test_seed():
    def reset_states(seed, **kwargs):
        mdp = MiniGrid('MiniGrid-DoorKey-5x5-v0', **kwargs)
        mdp.seed(seed)
        return [mdp.reset()[0] for _ in range(3)]

    first_run = reset_states(5)
    second_run = reset_states(5)
    other_run = reset_states(6)
    fixed_run = reset_states(6, fixed_seed=5)

    assert all(np.array_equal(first, second) for first, second in zip(first_run, second_run))
    assert not all(np.array_equal(first, other) for first, other in zip(first_run, other_run))
    assert all(np.array_equal(fixed_run[0], state) for state in fixed_run[1:])


def test_goal():
    mdp = MiniGrid('MiniGrid-Empty-5x5-v0', fixed_seed=1)
    mdp.reset()
    for action in [2, 2, 1, 2, 2]:
        _, reward, absorbing, _ = mdp.step(np.array([action]))

    assert reward == 1. and absorbing
