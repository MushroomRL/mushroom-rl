import os

import numpy as np
import pytest

from mushroom_rl.environments import PyBullet


def make_scene(actuation_spec, observation_spec, **kwargs):
    pybullet_data = pytest.importorskip('pybullet_data')

    class Scene(PyBullet):
        def reward(self, state, action, next_state, absorbing):
            return 0.

        def is_absorbing(self, state):
            return False

    path = pybullet_data.getDataPath()
    files = {os.path.join(path, 'plane.urdf'): dict(),
             os.path.join(path, 'cartpole.urdf'): dict(basePosition=[0, 2, 1]),
             os.path.join(path, 'cube_small.urdf'): dict(basePosition=[0, 0, .3]),
             os.path.join(path, 'mjcf', 'cylinder.xml'): dict(),
             os.path.join(path, 'plane_stadium.sdf'): dict()}

    return Scene(files, actuation_spec, observation_spec, .99, 100, **kwargs)


def test_pybullet_env():
    pybullet = pytest.importorskip('pybullet')
    from mushroom_rl.utils.pybullet import PyBulletObservationType

    mdp = make_scene([('slider_to_cart', pybullet.VELOCITY_CONTROL)],
                     [('slider_to_cart', PyBulletObservationType.JOINT_VEL),
                      ('cube.urdf<->plane', PyBulletObservationType.CONTACT_FLAG)], enforce_joint_velocity_limits=True)

    assert list(mdp._model_map) == ['plane', 'physics', 'cube.urdf', '', 'floor_obj']
    assert np.array_equal(mdp.info.action_space.low, [-5.]) and np.array_equal(mdp.info.action_space.high, [5.])
    assert np.array_equal(mdp.info.observation_space.low, [-5., 0.])
    assert np.array_equal(mdp.info.observation_space.high, [5., 1.])

    mdp.reset()
    contacts = [mdp.step(np.ones(1))[0][-1] for _ in range(100)]

    assert contacts[0] == 0. and contacts[-1] == 1.
    assert mdp.render(record=True).shape == (500, 500, 3)
    assert mdp.render() is None

    mdp.seed(3)
    first = np.random.rand()
    mdp.seed(3)

    assert np.random.rand() == first

    mdp.stop()

    with pytest.raises(NotImplementedError):
        PyBullet.reward(mdp, None, None, None, None)
    with pytest.raises(NotImplementedError):
        PyBullet.is_absorbing(mdp, None)
    with pytest.raises(NotImplementedError):
        make_scene([('slider_to_cart', pybullet.PD_CONTROL)], [('slider_to_cart', PyBulletObservationType.JOINT_POS)])
    with pytest.raises(RuntimeError):
        make_scene([('slider_to_cart', pybullet.TORQUE_CONTROL)], [('slider_to_cart', 'joint_pos')])
