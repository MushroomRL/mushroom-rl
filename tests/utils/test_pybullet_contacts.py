import pytest

from mushroom_rl.utils.pybullet.contacts import ContactHelper


class StubClient:
    def __init__(self, points):
        self._points = points

    def getContactPoints(self, model_a, model_b):
        return self._points.get((model_a, model_b), [])


def test_contact_helper_empty():
    helper = ContactHelper(StubClient({}), [], {}, {})
    helper.compute_contacts()

    assert helper._contact_list == []


def test_contact_helper_build_and_compute():
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


def test_contact_helper_order_contact():
    assert ContactHelper._order_contact(2, 5, 0, 1) == (2, 5, 0, 1)
    assert ContactHelper._order_contact(5, 2, 0, 1) == (2, 5, 1, 0)


def test_contact_helper_get_link_ids():
    link_map = {'paddle': (4, 0)}
    model_map = {'puck': 2}

    assert ContactHelper._get_link_ids('paddle', link_map, model_map) == (4, 0)
    assert ContactHelper._get_link_ids('puck', link_map, model_map) == (2, -1)


def test_contact_helper_real_pybullet():
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
