import numpy as np
import pytest

from helper.envs import constant_rollout

pytest.importorskip("mujoco")

from mushroom_rl.environments.mujoco_envs.air_hockey import (  # noqa: E402
    AirHockeyDefend, AirHockeyHit, AirHockeyPrepare, AirHockeyRepel, double)


@pytest.mark.parametrize('env, n_actions, name', [(double.AirHockeyDouble, 6, 'double'), (AirHockeyDefend, 3, 'defend'),
                                                  (AirHockeyHit, 3, 'hit'), (AirHockeyPrepare, 3, 'prepare'),
                                                  (AirHockeyRepel, 3, 'repel')])
def test_air_hockey(env, n_actions, name):
    observations = constant_rollout(env(), np.array([1] * n_actions))

    assert np.allclose(observations, np.load(f'tests/environments/mujoco_envs/data/test_air_hockey_{name}.npy'))


def reward_variants():
    return [pytest.param('prepare_side', AirHockeyPrepare, dict(random_init=True), id='prepare_side'),
            pytest.param('prepare_bottom', AirHockeyPrepare, dict(random_init=True, sub_problem='bottom'),
                         id='prepare_bottom'),
            pytest.param('hit_left', AirHockeyHit, dict(random_init=True, init_robot_state='left'), id='hit_left'),
            pytest.param('hit_random', AirHockeyHit, dict(init_robot_state='random'), id='hit_random'),
            pytest.param('defend_noise', AirHockeyDefend, dict(random_init=True, env_noise=True, obs_noise=True),
                         id='defend_noise'),
            pytest.param('repel', AirHockeyRepel, dict(random_init=True), id='repel'),
            pytest.param('double_noise', double.AirHockeyDouble, dict(env_noise=True, obs_noise=True),
                         id='double_noise')]


def reward_rollout(mdp, n_steps=100):
    np.random.seed(0)
    mdp.reset()
    rewards = list()
    for _ in range(n_steps):
        observation, reward, absorbing, _ = mdp.step(np.random.uniform(mdp.info.action_space.low,
                                                                       mdp.info.action_space.high))
        rewards.append(reward)

        assert not absorbing

    return np.array(rewards, dtype=float), observation


@pytest.mark.parametrize('name, env, params', reward_variants())
def test_rewards(name, env, params):
    rewards, observation = reward_rollout(env(**params))

    assert np.allclose(rewards, np.load(f'tests/environments/mujoco_envs/data/test_rewards_{name}.npy'))
    assert np.allclose(observation, np.load(f'tests/environments/mujoco_envs/data/test_rewards_{name}_obs.npy'))


def branch_cases():
    # (has_hit, has_bounce, puck position, puck linear velocity, absorbing)
    shared = [(False, False, [-.5, .3], [.1, 0.], False), (False, True, [-.5, .3], [.1, 0.], False),
              (True, False, [.1, .2], [.5, 0.], False), (True, False, [-.5, .005], [.5, 0.], False),
              (False, False, [5., 0.], [0., 0.], True)]
    return [pytest.param('prepare_side', AirHockeyPrepare, dict(),
                         shared + [(False, False, [-.5, .2], [.1, 0.], True),
                                   (True, False, [-.5, .2], [.1, 0.], False)], id='prepare_side'),
            pytest.param('prepare_bottom', AirHockeyPrepare, dict(sub_problem='bottom'),
                         shared + [(False, False, [-.5, .2], [.1, 0.], True),
                                   (True, False, [-.75, .2], [.1, .3], False),
                                   (True, False, [-.3, .2], [.1, 0.], False)], id='prepare_bottom'),
            pytest.param('hit', AirHockeyHit, dict(),
                         shared + [(True, False, [1.08, 0.], [1., 0.], True),
                                   (True, False, [.8, .05], [1.5, 0.], False),
                                   (True, False, [.3, .05], [1.5, 0.], False)], id='hit'),
            pytest.param('defend', AirHockeyDefend, dict(),
                         shared + [(False, False, [-1.08, 0.], [1., 0.], True),
                                   (True, False, [-.6, .1], [.1, 0.], False),
                                   (True, False, [-.2, .1], [.1, 0.], False)], id='defend'),
            pytest.param('repel', AirHockeyRepel, dict(),
                         shared + [(False, False, [-1.08, 0.], [1., 0.], True),
                                   (True, False, [.95, .1], [1., 0.], False),
                                   (True, False, [.3, .1], [2., 0.], False)], id='repel')]


def evaluate_branches(mdp, cases):
    np.random.seed(0)
    obs, _ = mdp.reset()
    values = list()
    for has_hit, has_bounce, position, velocity, absorbing in cases:
        mdp.has_hit, mdp.has_bounce = has_hit, has_bounce
        state = obs.copy()
        mdp.obs_helper.get_from_obs(state, 'puck_pos')[:] = position
        mdp.obs_helper.get_from_obs(state, 'puck_vel')[1:] = velocity
        values += [mdp.reward(obs, np.ones(3), state, absorbing), float(mdp.is_absorbing(state))]

    return np.array(values)


@pytest.mark.parametrize('name, env, params, cases', branch_cases())
def test_reward_branches(name, env, params, cases):
    values = evaluate_branches(env(**params), cases)

    assert np.allclose(values, np.load(f'tests/environments/mujoco_envs/data/test_reward_branches_{name}.npy'))


def test_double():
    mdp = double.AirHockeyDouble()
    obs, _ = mdp.reset()
    position, velocity, angular_velocity = mdp.get_puck(obs)

    assert np.array_equal(position, obs[:2]) and np.array_equal(velocity, obs[3:5]) and angular_velocity == obs[2]

    mdp.robot_1_hit = mdp.robot_2_hit = mdp.has_bounce = True
    mdp.step(np.zeros(6))

    assert mdp.robot_1_hit and mdp.robot_2_hit and mdp.has_bounce

    for robot, hits in [(1, (True, False)), (2, (False, True))]:
        mdp = double.AirHockeyDouble()
        mdp.reset()
        ee_pos, ee_vel = mdp.get_ee(robot)

        assert np.array_equal(ee_pos, mdp._read_data(f'robot_{robot}/ee_pos'))
        assert np.array_equal(ee_vel, mdp._read_data(f'robot_{robot}/ee_vel'))
        assert np.allclose(ee_pos, [-.81 if robot == 1 else .81, 0., 0.], atol=1e-3)

        puck = mdp._read_data('puck_pos').copy()
        puck[:2] = ee_pos[:2]
        mdp._write_data('puck_pos', puck)
        mdp.step(np.zeros(6))

        assert (mdp.robot_1_hit, mdp.robot_2_hit) == hits
