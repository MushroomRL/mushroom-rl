import numpy as np

from mushroom_rl.environments import LQR
from mushroom_rl.solvers.lqr import solve_lqr_linear, _compute_riccati_rhs


def test_lqr_solver_linear():
    lqr = LQR.generate(3)
    k = solve_lqr_linear(lqr)

    k_test = np.array(
        [[0.89908343, 0., 0.],
         [0., 0.24025307, 0.],
         [0., 0., 0.24025307]]
    )

    assert np.allclose(k, k_test)
