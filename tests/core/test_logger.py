import numpy as np
from pytest import importorskip
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


def test_default_wandb_kwargs():
    kwargs = Logger.default_wandb_kwargs('proj', config={'a': 1}, name='run', group='g')

    assert kwargs['project'] == 'proj'
    assert kwargs['config'] == {'a': 1}
    assert kwargs['name'] == 'run'
    assert kwargs['group'] == 'g'
    assert kwargs['entity'] is None
    assert kwargs['tags'] is None
    assert kwargs['mode'] == 'online'

    default_kwargs = Logger.default_wandb_kwargs('proj')
    assert default_kwargs['config'] == dict()


def test_logger_unified_log(tmpdir):
    logger = Logger('test_log', results_dir=tmpdir)

    assert not logger.wandb_active

    logger.log(x=1.0)
    assert not (tmpdir / 'test_log' / 'x.npy').exists()
    logger.advance_step()

    logger_numpy = Logger('test_log_numpy', results_dir=tmpdir, force_numpy=True)

    logger_numpy.log(y=2.0)
    y = np.load(str(tmpdir / 'test_log_numpy' / 'y.npy'))
    assert np.array_equal(y, np.array([2.0]))


def test_logger_no_results_dir():
    logger = Logger('test_log', results_dir=None, force_numpy=True)

    assert not logger.wandb_active

    logger.log(x=1.0)
    logger.log_wandb(x=1.0)
    logger.advance_step()


def test_wandb_offline(tmpdir):
    importorskip('wandb')

    wandb_kwargs = Logger.default_wandb_kwargs('test_project', config={'lr': 0.1},
                                               mode='offline', dir=str(tmpdir))
    logger = Logger('wandb_test', results_dir=None, wandb_kwargs=wandb_kwargs)

    assert logger.wandb_active
    assert logger._wandb_step == 0

    logger.log(actor_loss=0.5)
    logger.advance_step()
    assert logger._wandb_step == 1
    logger.log(actor_loss=0.25)

    assert logger._wandb_run.summary['actor_loss'] in (0.5, 0.25)

    logger._wandb_run.finish()
