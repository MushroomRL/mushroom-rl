import numpy as np

from mushroom.environments.car_on_hill import CarOnHill
from mushroom.solvers.car_on_hill import solve_car_on_hill


def test_car_on_hill():
    mdp = CarOnHill()

    states = np.array([[-.5, 0], [0., 0.], [.5, 0.]])
    actions = np.array([[0], [1], [0]])
    q = solve_car_on_hill(mdp, states, actions, .95, 10)
    q_test = np.array([-0.6302494097246091, -0.6302494097246091,
                       -0.6302494097246091])

    assert np.allclose(q, q_test)
