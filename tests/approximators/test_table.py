import numpy as np

from mushroom_rl.approximators.table import Table


def test_table():
    table = Table((3, 2))
    table.fit((1, 0), 5.)

    assert np.array_equal(table.table, [[0., 0.], [5., 0.], [0., 0.]])
    assert np.array_equal(table.predict(np.array([[1], [2]]), np.array([[0], [1]])), [5., 0.])
    assert table.predict(np.array([1]), np.array([0])) == 5.

    single = Table((1,))
    single[np.array([0])] = 3.

    assert single[np.array([0])] == 3. and np.array_equal(single.table, [3.])
