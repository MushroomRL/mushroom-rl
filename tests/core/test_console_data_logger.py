import re

import numpy as np

from mushroom_rl.core.logger.console_logger import ConsoleLogger
from mushroom_rl.core.logger.data_logger import DataLogger


def read_log(path):
    return re.sub(r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} ', '', path.read_text())


def test_console_logger(tmp_path, monkeypatch):
    logger = ConsoleLogger('exp', suffix='-seed1', log_dir=tmp_path)
    logger.debug('a debug message')
    logger.info('an info message')
    logger.warning('a warning message')
    logger.error('an error message')
    logger.critical('a critical message')
    logger.strong_line()
    logger.weak_line()
    logger.strong_line(debug=True, length=10)
    logger.weak_line(debug=True)
    logger.weak_line(length=5)
    logger.epoch_info(3, J=1.5, R=2.5)

    assert read_log(tmp_path / 'exp-seed1.log') == (
        '[DEBUG] a debug message\n'
        '[INFO] an info message\n'
        '[WARNING] a warning message\n'
        '[ERROR] an error message\n'
        '[CRITICAL] a critical message\n'
        '[INFO] ################################################################################\n'
        '[INFO] --------------------------------------------------------------------------------\n'
        '[DEBUG] ##########\n'
        '[DEBUG] --------------------------------------------------------------------------------\n'
        '[INFO] -----\n'
        '[INFO] Epoch 3 | J: 1.5 R: 2.5\n'
    )

    monkeypatch.chdir(tmp_path)
    ConsoleLogger('no_dir_logger').info('no file created')

    assert sorted(path.name for path in tmp_path.iterdir()) == ['exp-seed1.log']


def test_data_logger(tmp_path):
    logger = DataLogger(tmp_path)
    for i in range(5):
        logger.log_numpy(reward=float(i), steps=i * 10)
    weights = np.array([[1., 2.], [3., 4.]])
    logger.log_numpy_array(weights=weights)

    assert logger.path == tmp_path
    assert np.array_equal(np.load(str(tmp_path / 'reward.npy')), np.arange(5, dtype=float))
    assert np.array_equal(np.load(str(tmp_path / 'steps.npy')), np.arange(5) * 10)
    assert np.array_equal(np.load(str(tmp_path / 'weights.npy')), weights)

    logger = DataLogger(tmp_path, suffix='-1')
    for i in range(3):
        logger.log_numpy(val=float(i))
    DataLogger(tmp_path, suffix='-1', append=True).log_numpy(val=3.)

    assert np.array_equal(np.load(str(tmp_path / 'val-1.npy')), np.arange(4, dtype=float))
