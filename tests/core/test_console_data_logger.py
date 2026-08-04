import re
import numpy as np
from mushroom_rl.core.logger.console_logger import ConsoleLogger
from mushroom_rl.core.logger.data_logger import DataLogger


def _strip_timestamps(content):
    return re.sub(r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} ', '', content)


def test_console_logger_all_levels(tmp_path):
    logger = ConsoleLogger('test_levels', log_dir=tmp_path)
    logger.debug('a debug message')
    logger.info('an info message')
    logger.warning('a warning message')
    logger.error('an error message')
    logger.critical('a critical message')

    content = _strip_timestamps((tmp_path / 'test_levels.log').read_text())
    assert content == (
        '[DEBUG] a debug message\n'
        '[INFO] an info message\n'
        '[WARNING] a warning message\n'
        '[ERROR] an error message\n'
        '[CRITICAL] a critical message\n'
    )


def test_console_logger_lines(tmp_path):
    logger = ConsoleLogger('test_lines', log_dir=tmp_path)
    logger.strong_line()
    logger.weak_line()

    content = _strip_timestamps((tmp_path / 'test_lines.log').read_text())
    assert content == (
        '[INFO] ################################################################################\n'
        '[INFO] --------------------------------------------------------------------------------\n'
    )


def test_console_logger_lines_debug_and_length(tmp_path):
    logger = ConsoleLogger('test_lines_debug', log_dir=tmp_path)
    logger.strong_line(debug=True, length=10)
    logger.weak_line(debug=True)
    logger.weak_line(length=5)

    content = _strip_timestamps((tmp_path / 'test_lines_debug.log').read_text())
    assert content == (
        '[DEBUG] ##########\n'
        '[DEBUG] --------------------------------------------------------------------------------\n'
        '[INFO] -----\n'
    )


def test_console_logger_epoch_info(tmp_path):
    logger = ConsoleLogger('test_epoch', log_dir=tmp_path)
    logger.epoch_info(3, J=1.5, R=2.5)

    content = _strip_timestamps((tmp_path / 'test_epoch.log').read_text())
    assert content == '[INFO] Epoch 3 | J: 1.5 R: 2.5\n'


def test_console_logger_suffix(tmp_path):
    logger = ConsoleLogger('exp', suffix='-seed1', log_dir=tmp_path)
    logger.info('with suffix')
    assert (tmp_path / 'exp-seed1.log').exists()


def test_console_logger_no_dir():
    logger = ConsoleLogger('no_dir_logger')
    logger.info('no file created')


def test_data_logger_log_numpy(tmp_path):
    dl = DataLogger(tmp_path)
    for i in range(5):
        dl.log_numpy(reward=float(i), steps=i * 10)
    assert np.array_equal(np.load(str(tmp_path / 'reward.npy')), np.arange(5, dtype=float))
    assert np.array_equal(np.load(str(tmp_path / 'steps.npy')), np.arange(5) * 10)


def test_data_logger_log_numpy_array(tmp_path):
    dl = DataLogger(tmp_path)
    arr = np.array([[1., 2.], [3., 4.]])
    dl.log_numpy_array(weights=arr)
    assert np.array_equal(np.load(str(tmp_path / 'weights.npy')), arr)


def test_data_logger_suffix(tmp_path):
    dl = DataLogger(tmp_path, suffix='-1')
    dl.log_numpy(reward=1.)
    assert (tmp_path / 'reward-1.npy').exists()


def test_data_logger_append(tmp_path):
    dl = DataLogger(tmp_path, suffix='-1')
    for i in range(3):
        dl.log_numpy(val=float(i))
    dl2 = DataLogger(tmp_path, suffix='-1', append=True)
    dl2.log_numpy(val=3.)
    assert np.array_equal(np.load(str(tmp_path / 'val-1.npy')), np.arange(4, dtype=float))


def test_data_logger_path(tmp_path):
    dl = DataLogger(tmp_path)
    assert dl.path == tmp_path
