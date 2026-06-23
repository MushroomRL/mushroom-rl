import numpy as np
import pytest

from mushroom_rl.rl_utils.eligibility_trace import EligibilityTrace, ReplacingTrace, AccumulatingTrace


def test_replacing_trace():
    trace = EligibilityTrace((2, 2), 'replacing')
    assert isinstance(trace, ReplacingTrace)

    trace.update(0, 1)
    trace.update(0, 1)

    table_test = np.array([[0., 1.], [0., 0.]])
    assert np.allclose(trace.table, table_test)

    trace.reset()
    assert np.allclose(trace.table, np.zeros((2, 2)))


def test_accumulating_trace():
    trace = EligibilityTrace((2, 2), 'accumulating')
    assert isinstance(trace, AccumulatingTrace)

    trace.update(0, 1)
    trace.update(0, 1)

    table_test = np.array([[0., 2.], [0., 0.]])
    assert np.allclose(trace.table, table_test)

    trace.reset()
    assert np.allclose(trace.table, np.zeros((2, 2)))


def test_unknown_trace():
    with pytest.raises(ValueError):
        EligibilityTrace((2, 2), 'unknown')
