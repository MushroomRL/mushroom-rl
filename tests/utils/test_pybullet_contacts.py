import os

import numpy as np
import pytest

from mushroom_rl.environments import PyBullet
from mushroom_rl.utils.pybullet.contacts import ContactHelper


class StubClient:
    def __init__(self, points):
        self._points = points

    def getContactPoints(self, model_a, model_b):
        return self._points.get((model_a, model_b), [])


def test_contact_helper():
    helper = ContactHelper(StubClient({}), [], {}, {})
    helper.compute_contacts()

    assert helper._contact_list == []

    model_map = {'puck': 2}
    link_map = {'mallet_tip': (5, 0), 'mallet_base': (5, 1), 'table_surf': (3, 0), 'table_rim': (3, 1)}
    contacts = ['mallet_tip<->table_surf', 'mallet_base<->table_rim', 'puck<->table_surf',
                'mallet_tip<->table_surf']

    point = (0, 3, 5, 0, 0, 'data')
    client = StubClient({(3, 5): [point], (2, 3): []})

    helper = ContactHelper(client, contacts, model_map, link_map)

    assert ((3, 5), [(0, 0), (1, 1)]) in helper._contact_list
    assert ((2, 3), [(-1, 0)]) in helper._contact_list

    helper.compute_contacts()

    assert helper.get_contact('mallet_tip<->table_surf') == point
    assert helper.get_contact('mallet_base<->table_rim') is None
    assert helper.get_contact('puck<->table_surf') is None

    assert ContactHelper._order_contact(2, 5, 0, 1) == (2, 5, 0, 1)
    assert ContactHelper._order_contact(5, 2, 0, 1) == (2, 5, 1, 0)

    link_map = {'paddle': (4, 0)}
    model_map = {'puck': 2}

    assert ContactHelper._get_link_ids('paddle', link_map, model_map) == (4, 0)
    assert ContactHelper._get_link_ids('puck', link_map, model_map) == (2, -1)


def test_real_pybullet():
    p = pytest.importorskip('pybullet')

    p.connect(p.DIRECT)
    try:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
        box_a = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, basePosition=[0, 0, 0])
        box_b = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col, basePosition=[0.2, 0, 0])
        box_far = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col, basePosition=[100, 0, 0])
        p.stepSimulation()

        model_map = {'box_a': box_a, 'box_b': box_b, 'box_far': box_far}
        helper = ContactHelper(p, ['box_a<->box_b', 'box_a<->box_far'], model_map, {})
        helper.compute_contacts()

        touching = helper.get_contact('box_a<->box_b')
        assert touching is not None
        assert len(touching) == 14
        assert touching[3:5] == (-1, -1)

        assert helper.get_contact('box_a<->box_far') is None
    finally:
        p.disconnect()


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
