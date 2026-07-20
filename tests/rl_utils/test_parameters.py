import numpy as np

from mushroom_rl.rl_utils.parameters import Parameter, LinearParameter, DecayParameter
from mushroom_rl.rl_utils.parameters import VarianceIncreasingParameter
from mushroom_rl.policy import EpsGreedy, Mellowmax


class FakeLogger:
    def __init__(self):
        self.calls = list()

    def log_training(self, prefix=None, **kwargs):
        self.calls.append({(prefix + '/' + name if prefix else name): value for name, value in kwargs.items()})

    def advance_step(self):
        pass


def test_scalar_parameter_logs_on_update():
    logger = FakeLogger()
    p = Parameter(0.5)
    p.set_logger(logger, 'lr')

    value = p()

    assert np.isclose(value, 0.5)
    assert logger.calls == [{'lr/value': 0.5}]


def test_scalar_parameter_default_label():
    logger = FakeLogger()
    p = Parameter(0.3)
    p.set_logger(logger)

    p()

    assert logger.calls == [{'value': 0.3}]


def test_tabular_parameter_not_logged_by_default():
    logger = FakeLogger()
    p = Parameter(0.5, shape=(3, 3))
    p.set_logger(logger, 'lr')

    p(1, 2)

    assert logger.calls == []


def test_tabular_parameter_logged_when_forced():
    logger = FakeLogger()
    p = Parameter(0.5, shape=(3, 3), log_full=True)
    p.set_logger(logger, 'lr')

    p(1, 2)

    assert logger.calls == [{'lr/value': 0.5}]


def test_degenerate_table_parameter_is_logged():
    logger = FakeLogger()
    p = Parameter(0.5, shape=(1, 1))
    p.set_logger(logger, 'lr')

    p()

    assert logger.calls == [{'lr/value': 0.5}]


def test_linear_parameter_logs_post_update_value():
    logger = FakeLogger()
    p = LinearParameter(1.0, threshold_value=0.0, n=10)
    p.set_logger(logger, 'eps')

    p()

    assert len(logger.calls) == 1
    assert np.isclose(logger.calls[0]['eps/value'], 0.9)


def test_decay_parameter_logs_post_update_value():
    logger = FakeLogger()
    p = DecayParameter(1.0, exp=1.0)
    p.set_logger(logger, 'eps')

    p()
    p()

    assert len(logger.calls) == 2
    assert np.isclose(logger.calls[0]['eps/value'], 1.0)
    assert np.isclose(logger.calls[1]['eps/value'], 0.5)


def test_parameter_without_logger_does_not_raise():
    p = Parameter(0.5)
    value = p()

    assert p._logger is None
    assert np.isclose(value, 0.5)


def test_variance_parameter_logs_on_update():
    logger = FakeLogger()
    p = VarianceIncreasingParameter(value=0.5, tol=1.)
    p.set_logger(logger, 'lr')

    p.update(0, target=1.0)

    assert len(logger.calls) == 1
    assert np.isclose(logger.calls[0]['lr/value'], 0.5)


def test_variance_parameter_with_shape_tracks_indices_independently():
    p = VarianceIncreasingParameter(value=1., tol=1., shape=(2,))

    p.update(0, target=1.0)
    p.update(0, target=2.0)
    p.update(0, target=1.0)

    p.update(1, target=5.0)

    assert np.isclose(p.get_value(0), 0.3333333333333333)
    assert np.isclose(p.get_value(1), 1.0)


def test_eps_greedy_forwards_logger():
    logger = FakeLogger()
    pi = EpsGreedy(Parameter(0.1))
    pi.set_logger(logger)

    pi.update()

    assert len(logger.calls) == 1
    assert np.isclose(logger.calls[0]['policy/epsilon'], 0.1)


def test_mellowmax_set_logger_does_not_log():
    logger = FakeLogger()
    pi = Mellowmax(Parameter(3.))
    pi.set_logger(logger)

    assert logger.calls == []


def test_parameter_serialization_roundtrip(tmpdir):
    p = Parameter(0.5, shape=(2, 2), log_full=True)
    path = str(tmpdir / 'param.msh')
    p.save(path)

    p2 = Parameter.load(path)

    assert p2._log_full is True
    assert p2._logger is None
    p2(0, 0)
