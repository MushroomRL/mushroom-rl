import numpy as np
import pytest

from helper.envs import constant_rollout

from mushroom_rl.core import Environment

pytest.importorskip("pybullet")

from mushroom_rl.utils.pybullet import PyBulletObservationType  # noqa: E402


@pytest.mark.parametrize('env, kwargs, name', [('AirHockeyDefendBullet', dict(), 'defend'),
                                               ('AirHockeyHitBullet', dict(), 'hit'),
                                               ('AirHockeyPrepareBullet', dict(), 'prepare'),
                                               ('AirHockeyRepelBullet', dict(), 'repel'),
                                               ('AirHockeyPrepareBullet', dict(sub_problem='bottom'), 'prepare_bottom'),
                                               ('AirHockeyHitBullet', dict(init_robot_state='left'), 'hit_left')])
def test_air_hockey(env, kwargs, name):
    observations = constant_rollout(Environment.make(env, **kwargs), np.array([1] * 3))

    assert np.allclose(observations, np.load(f"tests/environments/pybullet_envs/data/test_air_hockey_{name}.npy"))


def branch_cases():
    # (has_hit, has_bounce, puck position, puck linear velocity, end effector position or None, absorbing)
    shared = [(False, False, [-.5, .3], [.1, 0.], None, False), (False, True, [-.5, .3], [.1, 0.], None, False),
              (True, False, [.1, .2], [.5, 0.], None, False), (True, False, [-.5, .005], [.5, 0.], None, False),
              (False, False, [5., 0.], [0., 0.], None, True)]
    boundary = [(False, False, [1.08, 0.], [1., 0.], [2., 0.], True),
                (False, False, [-.5, .3], [.1, 0.], [0., 2.], True)]
    return [pytest.param('prepare_side', 'AirHockeyPrepareBullet', dict(),
                         shared + [(False, False, [-.5, .2], [.1, 0.], None, True),
                                   (True, False, [-.5, .2], [.1, 0.], None, False)], id='prepare_side'),
            pytest.param('prepare_bottom', 'AirHockeyPrepareBullet', dict(sub_problem='bottom'),
                         shared + [(False, False, [-.5, .2], [.1, 0.], None, True),
                                   (True, False, [-.75, .2], [.1, .3], None, False),
                                   (True, False, [-.3, .2], [.1, 0.], None, False)], id='prepare_bottom'),
            pytest.param('hit', 'AirHockeyHitBullet', dict(),
                         shared + [(True, False, [1.08, 0.], [1., 0.], None, True),
                                   (True, False, [.8, .05], [1.5, 0.], None, False),
                                   (True, False, [.3, .05], [1.5, 0.], None, False),
                                   (True, False, [.3, .05], [0., 0.], None, False)], id='hit'),
            pytest.param('hit_boundary', 'AirHockeyHitBullet', dict(table_boundary_terminate=True), shared + boundary,
                         id='hit_boundary'),
            pytest.param('defend', 'AirHockeyDefendBullet', dict(),
                         shared + [(False, False, [-1.08, 0.], [1., 0.], None, True),
                                   (True, False, [-.6, .1], [.1, 0.], None, False),
                                   (True, False, [-.2, .1], [.1, 0.], None, False)], id='defend'),
            pytest.param('repel', 'AirHockeyRepelBullet', dict(),
                         shared + [(False, False, [-1.08, 0.], [1., 0.], None, True),
                                   (True, False, [.95, .1], [1., 0.], None, False),
                                   (True, False, [.3, .1], [2., 0.], None, False)], id='repel')]


def evaluate_branches(mdp, cases):
    np.random.seed(0)
    mdp.reset()
    base = mdp._state.copy()
    position = mdp.get_sim_state_index('puck', PyBulletObservationType.BODY_POS)[:2]
    velocity = mdp.get_sim_state_index('puck', PyBulletObservationType.BODY_LIN_VEL)[:2]
    ee = mdp.get_sim_state_index('planar_robot_1/link_striker_ee', PyBulletObservationType.LINK_POS)[:2]
    values = list()
    for has_hit, has_bounce, puck_position, puck_velocity, ee_position, absorbing in cases:
        state = base.copy()
        state[position] = puck_position
        state[velocity] = puck_velocity
        if ee_position is not None:
            state[ee] = ee_position
        mdp._state = state
        mdp.has_hit, mdp.has_bounce = has_hit, has_bounce
        values += [mdp.reward(state, np.ones(3), state, absorbing), float(mdp.is_absorbing(state))]

    return np.array(values)


@pytest.mark.parametrize('name, env, params, cases', branch_cases())
def test_reward_branches(name, env, params, cases):
    values = evaluate_branches(Environment.make(env, **params), cases)

    assert np.allclose(values, np.load(f'tests/environments/pybullet_envs/data/test_reward_branches_{name}.npy'))


def reward_variants():
    return [pytest.param('defend_random', 'AirHockeyDefendBullet', dict(random_init=True), id='defend_random'),
            pytest.param('repel_noise', 'AirHockeyRepelBullet',
                         dict(random_init=True, obs_noise=True), id='repel_noise'),
            pytest.param('hit_random', 'AirHockeyHitBullet', dict(random_init=True, init_robot_state='random'),
                         id='hit_random'),
            pytest.param('prepare_random', 'AirHockeyPrepareBullet', dict(random_init=True, sub_problem='bottom'),
                         id='prepare_random'),
            pytest.param('hit_position', 'AirHockeyHitBullet', dict(torque_control=False), id='hit_position')]


def reward_rollout(mdp, n_steps=100):
    np.random.seed(0)
    mdp.reset()
    rewards = list()
    for _ in range(n_steps):
        observation, reward, absorbing, _ = mdp.step(np.random.uniform(mdp.info.action_space.low,
                                                                       mdp.info.action_space.high))
        rewards.append(reward)
        if absorbing:
            break

    return np.array(rewards, dtype=float), observation


@pytest.mark.parametrize('name, env, params', reward_variants())
def test_rewards(name, env, params):
    rewards, observation = reward_rollout(Environment.make(env, **params))

    assert np.allclose(rewards, np.load(f'tests/environments/pybullet_envs/data/test_rewards_{name}.npy'))
    assert np.allclose(observation, np.load(f'tests/environments/pybullet_envs/data/test_rewards_{name}_obs.npy'))


def test_step_action_function():
    calls = list()

    def zero_action(state, action):
        calls.append(action)
        return np.zeros(3)

    wrapped = constant_rollout(Environment.make('AirHockeyHitBullet', step_action_function=zero_action),
                               np.ones(3), n_steps=5)
    reference = constant_rollout(Environment.make('AirHockeyHitBullet'), np.zeros(3), n_steps=5)

    assert len(calls) == 5 and np.array_equal(calls[0], np.ones(3))
    assert np.allclose(wrapped, reference)
