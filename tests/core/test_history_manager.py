import numpy as np

from mushroom_rl.core.array_backend import ArrayBackend
from mushroom_rl.core.history_manager import HistoryManager


def _make_manager(history_length=3, obs_shape=(2,)):
    backend = ArrayBackend.get_array_backend('numpy')
    return HistoryManager(history_length, obs_shape, float, backend, backend)


def test_history_length_property():
    hm = _make_manager(history_length=4)
    assert hm.history_length == 4


def test_single_assembly_and_shift():
    hm = _make_manager(history_length=3, obs_shape=(2,))
    hm.reset()

    stacked_1 = hm(np.array([1.0, 1.0]))
    assert stacked_1.shape == (3, 2)
    assert np.allclose(stacked_1, np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]))

    stacked_2 = hm(np.array([2.0, 2.0]))
    assert np.allclose(stacked_2, np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))


def test_single_reset_clears_buffer():
    hm = _make_manager(history_length=3, obs_shape=(2,))
    hm.reset()
    hm(np.array([1.0, 1.0]))
    hm(np.array([2.0, 2.0]))

    hm.reset()
    stacked = hm(np.array([5.0, 5.0]))
    assert np.allclose(stacked, np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0]]))


def test_vectorized_assembly():
    hm = _make_manager(history_length=3, obs_shape=(2,))
    hm.reset_vectorized(np.array([True, True]))

    stacked = hm(np.array([[1.0, 1.0], [2.0, 2.0]]))
    assert stacked.shape == (2, 3, 2)
    assert np.allclose(stacked[:, -1], np.array([[1.0, 1.0], [2.0, 2.0]]))
    assert np.allclose(stacked[:, :-1], 0.0)


def test_vectorized_masked_reset():
    hm = _make_manager(history_length=3, obs_shape=(2,))
    hm.reset_vectorized(np.array([True, True]))
    hm(np.array([[1.0, 1.0], [2.0, 2.0]]))
    hm(np.array([[3.0, 3.0], [4.0, 4.0]]))

    hm.reset_vectorized(np.array([True, False]))

    stacked = hm(np.array([[7.0, 7.0], [8.0, 8.0]]))
    assert np.allclose(stacked[0], np.array([[0.0, 0.0], [0.0, 0.0], [7.0, 7.0]]))
    assert np.allclose(stacked[1], np.array([[2.0, 2.0], [4.0, 4.0], [8.0, 8.0]]))


def test_vectorized_reset_does_not_corrupt_previously_returned_stack():
    hm = _make_manager(history_length=3, obs_shape=(2,))
    hm.reset_vectorized(np.array([True, True]))
    hm(np.array([[1.0, 1.0], [2.0, 2.0]]))

    stacked_1 = hm(np.array([[3.0, 3.0], [4.0, 4.0]]))
    expected_stacked_1 = stacked_1.copy()

    hm.reset_vectorized(np.array([True, False]))

    assert np.allclose(stacked_1, expected_stacked_1)


def test_single_call_does_not_corrupt_previously_returned_stack():
    hm = _make_manager(history_length=3, obs_shape=(2,))
    hm.reset()

    stacked_1 = hm(np.array([1.0, 1.0]))
    expected_stacked_1 = stacked_1.copy()

    hm(np.array([2.0, 2.0]))

    assert np.allclose(stacked_1, expected_stacked_1)
