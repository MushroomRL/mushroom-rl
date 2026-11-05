import numpy as np

from mushroom_rl.environments.car_on_hill import CarOnHill
from mushroom_rl.solvers.car_on_hill import solve_car_on_hill


def test_car_on_hill():
    mdp = CarOnHill()
    mdp._discrete_actions = np.array([-8., 8.])

    states = np.array([[-.5, 0], [0., 0.], [.5, 0.]])
    actions = np.array([[0], [1], [0]])
    q = solve_car_on_hill(mdp, states, actions, .95)
    q_test = np.array([0.5688000922764597, 0.48767497911552954,
                       0.5688000922764597])

    assert np.allclose(q, q_test)

    q = solve_car_on_hill(mdp, np.array([[.99, 1.], [-.99, -2.], [.5, 0.]]), np.array([[1], [0], [0]]), .95, max_k=2)

    assert np.allclose(q, [1., -1., -.95])
    assert np.allclose(solve_car_on_hill(mdp, np.array([[.9, 1.]]), np.array([[0]]), .95), [.95])
