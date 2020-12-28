import numpy as np
from mushroom_rl.core import Logger


def test_logger(tmpdir):
    logger_1 = Logger('test', seed=1, results_dir=tmpdir)
    logger_2 = Logger('test', seed=2, results_dir=tmpdir)

    for i in range(3):
        logger_1.log_numpy(a=i, b=2*i+1)
        logger_2.log_numpy(a=2*i+1, b=i)

    a_1 = np.load(str(tmpdir / 'test' / 'a-1.npy'))
    a_2 = np.load(str(tmpdir / 'test' / 'a-2.npy'))
    b_1 = np.load(str(tmpdir / 'test' / 'b-1.npy'))
    b_2 = np.load(str(tmpdir / 'test' / 'b-2.npy'))

    assert np.array_equal(a_1, np.arange(3))
    assert np.array_equal(b_2, np.arange(3))
    assert np.array_equal(a_1, b_2)
    assert np.array_equal(b_1, a_2)

    logger_1_bis = Logger('test', append=True, seed=1, results_dir=tmpdir)

    logger_1_bis.log_numpy(a=3, b=7)
    a_1 = np.load(str(tmpdir / 'test' / 'a-1.npy'))
    b_2 = np.load(str(tmpdir / 'test' / 'b-2.npy'))

    assert np.array_equal(a_1, np.arange(4))
    assert np.array_equal(b_2, np.arange(3))
