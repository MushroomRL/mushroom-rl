import numpy as np
from mushroom.environments.car_on_hill import CarOnHill


def test_car_on_hill():
    np.random.seed(1)
    mdp = CarOnHill()
    mdp.reset()
    for i in range(10):
        ns, r, ab, _ = mdp.step([np.random.randint(2)])
    ns_test = np.array([-0.29638141, -0.05527507])

    angle = mdp._angle(ns_test[0])
    angle_test = -1.141676764064636

    height = mdp._height(ns_test[0])
    height_test = 0.9720652903871763

    assert np.allclose(ns, ns_test)
    assert np.allclose(angle, angle_test)
    assert np.allclose(height, height_test)
