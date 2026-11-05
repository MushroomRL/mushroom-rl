import numpy as np
import pytest

from helper.envs import rollout, scalar_action

pytest.importorskip("mujoco")

from mushroom_rl.environments import Ant, HalfCheetah, Hopper, Walker2D  # noqa: E402


@pytest.mark.parametrize('env, name', [(Ant, 'ant'), (HalfCheetah, 'half_cheetah'), (Hopper, 'hopper'),
                                       (Walker2D, 'walker2d')])
def test_locomotion(env, name):
    np.random.seed(1)
    ns = rollout(env(), scalar_action)

    assert np.allclose(ns, np.load(f'tests/environments/mujoco_envs/data/test_locomotion_{name}.npy'))


def test_ant_contact_forces():
    np.random.seed(4)
    mdp = Ant(use_contact_forces=True, exclude_current_positions_from_observation=False, terminate_when_unhealthy=False)
    mdp.reset()
    rewards, contact_costs = list(), list()
    for _ in range(90):
        observation, reward, _, info = mdp.step(np.random.uniform(-1, 1, 8))
        rewards.append(reward)
        contact_costs.append(info['contact_cost'])

    expected = np.load('tests/environments/mujoco_envs/data/test_ant_contact_forces.npy')

    assert mdp.info.observation_space.shape == (35,)
    assert np.count_nonzero(contact_costs) > 0
    assert np.allclose(np.concatenate([rewards, contact_costs, observation]), expected)
