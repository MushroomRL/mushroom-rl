import numpy as np
import pytest

from mushroom_rl.core import Environment, MDPInfo, VectorizedEnvironment, register_env
from mushroom_rl.core.spaces import Box
from mushroom_rl.environments import LQR


def make_info():
    return MDPInfo(Box(-1., 1., (3,)), Box(-1., 1., (2,)), 0.9, 10)


def test_register():
    def make_env_class():
        class RegisterEnvTestEnv(Environment):
            pass

        return RegisterEnvTestEnv

    first = make_env_class()
    second = make_env_class()

    try:
        assert register_env(first) is first
        assert 'RegisterEnvTestEnv' in Environment.list_registered()
        assert register_env(second) is second
        assert Environment._registered_envs['RegisterEnvTestEnv'] is first
    finally:
        Environment._registered_envs.pop('RegisterEnvTestEnv', None)


def test_registry():
    base_classes = ['Environment', 'VectorizedEnvironment', 'MultiprocessEnvironment', 'MuJoCo', 'MultiMuJoCo',
                    'MuJoCoWarp', 'PyBullet', 'IsaacSim', 'QuadrupedIsaac', 'MiniGridBase', 'AirHockeyBase',
                    'AirHockeySingle', 'AirHockeyDouble', 'Panda', 'AirHockeyBaseBullet', 'AirHockeySingleBullet']
    environments = ['CarOnHill', 'CartPole', 'FiniteMDP', 'GridWorld', 'GridWorldVanHasselt', 'InvertedPendulum', 'LQR',
                    'PuddleWorld', 'Segway', 'ShipSteering', 'SimpleChain', 'Taxi']

    registered = Environment.list_registered()

    assert not set(base_classes) & set(registered)
    assert set(environments) <= set(registered)


def test_make():
    class MakeTestEnv(Environment):
        def __init__(self, *args, flag=False):
            self.args = args
            self.flag = flag
            super().__init__(make_info())

    try:
        register_env(MakeTestEnv)
        env = Environment.make('MakeTestEnv.first.second', 3, flag=True)

        assert isinstance(env, MakeTestEnv)
        assert env.args == ('first', 'second', 3)
        assert env.flag
    finally:
        Environment._registered_envs.pop('MakeTestEnv', None)

    env = Environment.make('LQR', dimensions=2)

    assert isinstance(env, LQR)
    assert env.info.shape == (2, 2)


def test_interface():
    info = make_info()
    env = Environment(info)

    assert info.shape == (3, 2)
    assert env.info is info
    assert env.full_name() == 'Environment'

    with pytest.warns(UserWarning):
        env.seed(1)
    with pytest.raises(NotImplementedError):
        env.reset()
    with pytest.raises(NotImplementedError):
        env.step(np.zeros(2))
    with pytest.raises(NotImplementedError):
        env.render()

    vectorized = VectorizedEnvironment(info, 2)
    env_mask = np.ones(2, dtype=bool)

    with pytest.raises(NotImplementedError):
        vectorized.reset_all(env_mask)
    with pytest.raises(NotImplementedError):
        vectorized.step_all(env_mask, np.zeros((2, 2)))
    with pytest.raises(NotImplementedError):
        vectorized.render_all(env_mask)


def test_vectorized_render():
    class FrameEnv(VectorizedEnvironment):
        def __init__(self, frame):
            self.frame = frame
            self.masks = list()
            super().__init__(make_info(), 3)

        def render_all(self, env_mask, record=False):
            self.masks.append(env_mask.copy())
            return self.frame if record else None

    frames = np.arange(3 * 2 * 2 * 3).reshape(3, 2, 2, 3)
    env = FrameEnv(frames)
    env.set_default_env(1)

    assert env.render() is None
    assert np.array_equal(env.render(record=True), frames[0])
    assert np.array_equal(env.masks[0], [False, True, False])

    env = FrameEnv(frames[1])

    assert np.array_equal(env.render(record=True), frames[1])
