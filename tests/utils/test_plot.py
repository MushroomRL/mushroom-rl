import numpy as np

from mushroom_rl.utils.plot import get_mean_and_confidence


def test_get_mean_and_confidence():
    data = np.array([[1.0, 2.0, 3.0],
                     [2.0, 3.0, 4.0],
                     [3.0, 4.0, 5.0],
                     [0.0, 1.0, 2.0]])

    mean, interval = get_mean_and_confidence(data)

    mean_test = np.array([1.5, 2.5, 3.5])
    interval_test = np.array([2.05426026, 2.05426026, 2.05426026])

    assert np.allclose(mean, mean_test)
    assert np.allclose(interval, interval_test)
