import json
import logging
import numpy as np
import torch.optim as optim
from pathlib import Path
from pytest import importorskip, raises, warns
from mushroom_rl.algorithms.value import QLearning
from mushroom_rl.core import Logger
from mushroom_rl.core.dataset import Dataset
from mushroom_rl.environments import SimpleChain
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter


def release_logger_name(log_id):
    logging.getLogger(log_id).handlers.clear()


def test_logger(tmpdir):
    logger_1 = Logger('test_logger', seed=1, results_dir=tmpdir)
    logger_2 = Logger('test_logger', seed=2, results_dir=tmpdir)

    for i in range(3):
        logger_1.log_numpy(a=i, b=2*i+1)
        logger_2.log_numpy(a=2*i+1, b=i)

    a_1 = np.load(str(tmpdir / 'test_logger' / 'a-1.npy'))
    a_2 = np.load(str(tmpdir / 'test_logger' / 'a-2.npy'))
    b_1 = np.load(str(tmpdir / 'test_logger' / 'b-1.npy'))
    b_2 = np.load(str(tmpdir / 'test_logger' / 'b-2.npy'))

    assert np.array_equal(a_1, np.arange(3))
    assert np.array_equal(b_2, np.arange(3))
    assert np.array_equal(a_1, b_2)
    assert np.array_equal(b_1, a_2)

    release_logger_name('test_logger-1')
    logger_1_bis = Logger('test_logger', append=True, seed=1, results_dir=tmpdir)

    logger_1_bis.log_numpy(a=3, b=7)
    a_1 = np.load(str(tmpdir / 'test_logger' / 'a-1.npy'))
    b_2 = np.load(str(tmpdir / 'test_logger' / 'b-2.npy'))

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


def test_logger_training_evaluation(tmpdir):
    logger = Logger('test_logger_training_evaluation', results_dir=tmpdir)

    assert not logger.wandb_active

    logger.log_training(x=1.0)
    assert not (tmpdir / 'test_logger_training_evaluation' / 'training' / 'x.npy').exists()
    logger.advance_step()

    logger.log_evaluation(0, J=5.0)
    J = np.load(str(tmpdir / 'test_logger_training_evaluation' / 'J.npy'))
    assert np.array_equal(J, np.array([5.0]))

    logger_numpy = Logger('test_log_numpy', results_dir=tmpdir, force_numpy=True)

    logger_numpy.log_training(y=2.0)
    y = np.load(str(tmpdir / 'test_log_numpy' / 'training' / 'y.npy'))
    assert np.array_equal(y, np.array([2.0]))


def test_logger_append_evaluation(tmpdir):
    logger = Logger('test_logger_append_evaluation', results_dir=tmpdir)

    for i in range(3):
        logger.log_evaluation(i, J=float(i))

    J = np.load(str(tmpdir / 'test_logger_append_evaluation' / 'J.npy'))
    assert np.array_equal(J, np.array([0.0, 1.0, 2.0]))

    release_logger_name('test_logger_append_evaluation')
    logger_append = Logger('test_logger_append_evaluation', results_dir=tmpdir, append=True)
    logger_append.log_evaluation(3, J=3.0)

    J = np.load(str(tmpdir / 'test_logger_append_evaluation' / 'J.npy'))
    assert np.array_equal(J, np.array([0.0, 1.0, 2.0, 3.0]))


def test_logger_append_training(tmpdir):
    logger = Logger('test_logger_append_training', results_dir=tmpdir, force_numpy=True)

    for i in range(3):
        logger.log_training(x=float(i), y=float(2 * i + 1))
        logger.advance_step()

    x = np.load(str(tmpdir / 'test_logger_append_training' / 'training' / 'x.npy'))
    assert np.array_equal(x, np.array([0.0, 1.0, 2.0]))

    release_logger_name('test_logger_append_training')
    logger_append = Logger('test_logger_append_training', results_dir=tmpdir, append=True, force_numpy=True)
    logger_append.log_training(x=3.0, y=7.0)

    x = np.load(str(tmpdir / 'test_logger_append_training' / 'training' / 'x.npy'))
    y = np.load(str(tmpdir / 'test_logger_append_training' / 'training' / 'y.npy'))
    assert np.array_equal(x, np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.array_equal(y, np.array([1.0, 3.0, 5.0, 7.0]))


def test_logger_no_results_dir():
    logger = Logger('test_log', results_dir=None, force_numpy=True)

    assert not logger.wandb_active

    logger.log_training(x=1.0)
    logger.log_wandb_training(x=1.0)
    logger.advance_step()


def test_logger_duplicated_name():
    logger = Logger('test_duplicated_name', results_dir=None)

    with raises(AssertionError):
        Logger('test_duplicated_name', results_dir=None)

    logger.info('the first logger still works')


def test_wandb_offline(tmpdir):
    importorskip('wandb')

    wandb_kwargs = Logger.default_wandb_kwargs('test_project', config={'lr': 0.1},
                                               mode='offline', dir=str(tmpdir))
    logger = Logger('wandb_test', results_dir=None, wandb_kwargs=wandb_kwargs)

    assert logger.wandb_active
    assert logger._n_fit == 0
    assert logger._wandb_run.group == 'wandb_test'

    logger.log_training(actor_loss=0.5)
    logger.advance_step()
    assert logger._n_fit == 1
    logger.log_training(actor_loss=0.25)
    logger.advance_step()
    logger.log_evaluation(1, J=10.0)

    assert logger._wandb_run.summary['training/actor_loss'] in (0.5, 0.25)

    logger.finish()
    assert not logger.wandb_active


def test_wandb_group_auto_set(tmpdir):
    importorskip('wandb')

    wandb_kwargs = Logger.default_wandb_kwargs('test_project', mode='offline', dir=str(tmpdir))
    logger = Logger('test_wandb_group_auto_set', results_dir=None, wandb_kwargs=wandb_kwargs)

    assert logger._wandb_run.group == 'test_wandb_group_auto_set'
    logger.finish()


def test_wandb_group_not_overridden(tmpdir):
    importorskip('wandb')

    wandb_kwargs = Logger.default_wandb_kwargs('test_project', mode='offline',
                                               dir=str(tmpdir), group='custom_group')
    logger = Logger('test_wandb_group_not_overridden', results_dir=None, wandb_kwargs=wandb_kwargs)

    assert logger._wandb_run.group == 'custom_group'
    logger.finish()


def test_wandb_seed(tmpdir):
    importorskip('wandb')

    wandb_kwargs = Logger.default_wandb_kwargs('test_project', mode='offline', dir=str(tmpdir))
    logger = Logger('test_wandb_seed', results_dir=None, seed=42, wandb_kwargs=wandb_kwargs)

    assert logger._wandb_run.config['seed'] == 42
    assert logger._wandb_run.name == 'test_wandb_seed_42'
    assert logger._wandb_run.group == 'test_wandb_seed'
    logger.finish()


def test_wandb_seed_name_not_overridden(tmpdir):
    importorskip('wandb')

    wandb_kwargs = Logger.default_wandb_kwargs('test_project', mode='offline',
                                               dir=str(tmpdir), name='custom_name')
    logger = Logger('test_wandb_seed_name_not_overridden', results_dir=None, seed=42, wandb_kwargs=wandb_kwargs)

    assert logger._wandb_run.config['seed'] == 42
    assert logger._wandb_run.name == 'custom_name'
    logger.finish()


def test_wandb_append_offline(tmpdir):
    importorskip('wandb')

    wandb_kwargs = Logger.default_wandb_kwargs('test_project', config={'lr': 0.1},
                                               mode='offline', dir=str(tmpdir))
    logger = Logger('wandb_append', results_dir=tmpdir, wandb_kwargs=wandb_kwargs)

    assert logger.wandb_active
    run_id = logger._wandb_run.id

    for i in range(5):
        logger.log_training(loss=float(i))
        logger.advance_step()

    assert logger._n_fit == 5
    logger.finish()

    wandb_kwargs_2 = Logger.default_wandb_kwargs('test_project', config={'lr': 0.1},
                                                 mode='offline', dir=str(tmpdir))
    release_logger_name('wandb_append')
    logger_append = Logger('wandb_append', results_dir=tmpdir, append=True,
                           wandb_kwargs=wandb_kwargs_2)

    assert logger_append.wandb_active
    assert logger_append._wandb_run.id == run_id
    assert logger_append._n_fit == 0  # offline mode cannot restore the run

    logger_append.log_training(loss=5.0)
    logger_append.advance_step()
    assert logger_append._n_fit == 1

    logger_append.finish()


def test_video_logger_lazy_creation(tmpdir):
    logger = Logger('test_video_logger_lazy_creation', results_dir=tmpdir)

    assert logger.video_recorder is None

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    logger.set_video_fps(30)
    logger.record_frame(frame)

    assert logger.video_recorder is not None

    logger.stop_recording()


def test_video_logger_ignores_missing_frames(tmpdir):
    logger = Logger('test_video_logger_ignores_missing_frames', results_dir=tmpdir)

    logger.set_video_fps(30)

    with warns(UserWarning, match='The environment drew nothing'):
        logger.record_frame(None)

    assert logger.video_recorder is None
    assert logger.stop_recording() is None


def test_video_logger_custom_recorder():
    class FrameRecorder:
        def __init__(self, fps=None):
            self.frames = []

        def __call__(self, frame):
            self.frames.append(frame)

        def stop(self):
            pass

    logger = Logger('test_video_logger_custom_recorder', results_dir=None, recorder_class=FrameRecorder, fps=30)

    frame_1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_2 = np.ones((100, 100, 3), dtype=np.uint8)

    logger.record_frame(frame_1)
    logger.record_frame(frame_2)

    assert len(logger.video_recorder.frames) == 2
    assert np.array_equal(logger.video_recorder.frames[0], frame_1)
    assert np.array_equal(logger.video_recorder.frames[1], frame_2)

    logger.stop_recording()


def test_video_logger_set_fps():
    class FrameRecorder:
        def __init__(self, fps=None):
            self.fps = fps

        def __call__(self, frame):
            pass

        def stop(self):
            pass

    logger = Logger('test_video_logger_set_fps', results_dir=None, recorder_class=FrameRecorder)

    logger.set_video_fps(30)
    logger.set_video_fps(60)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    logger.record_frame(frame)

    assert logger.video_recorder.fps == 30


def test_video_logger_video_path(tmpdir):
    logger = Logger('test_video_logger_video_path', results_dir=tmpdir, fps=30)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    logger.record_frame(frame)
    path = logger.stop_recording()

    video_dir = Path(tmpdir) / 'test_video_logger_video_path' / 'videos'
    assert video_dir.exists()

    videos = list(video_dir.rglob('*.mp4'))
    assert len(videos) == 1

    assert path == videos[0]
    assert logger.recorded_videos == [path]


def test_video_logger_append(tmpdir):
    logger = Logger('test_video_logger_append', results_dir=tmpdir, fps=30)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    logger.record_frame(frame)
    path = logger.stop_recording()

    release_logger_name('test_video_logger_append')
    logger_append = Logger('test_video_logger_append', results_dir=tmpdir, fps=30, append=True)
    assert logger_append.recorded_videos == [path]


def test_log_dataset(tmpdir):
    logger = Logger('test_log_dataset', results_dir=tmpdir)
    logger_seed = Logger('test_seed', seed=42, results_dir=tmpdir)

    states = np.arange(10).reshape(5, 2).astype(float)
    actions = np.arange(5).reshape(5, 1).astype(float)
    rewards = np.arange(5).astype(float)
    next_states = states + 1
    absorbing = np.zeros(5)
    last = np.array([0, 0, 0, 0, 1], dtype=float)

    dataset = Dataset.from_array(states, actions, rewards, next_states, absorbing, last)

    logger.log_dataset(dataset)
    logger_seed.log_dataset(dataset)

    loaded = Dataset.load(logger.path / 'dataset.msh')
    assert np.array_equal(loaded.state, states)
    assert np.array_equal(loaded.action, actions)
    assert np.array_equal(loaded.reward, rewards)

    assert (logger_seed.path / 'dataset-42.msh').exists()


def test_log_video_wandb_offline(tmpdir):
    importorskip('wandb')

    wandb_kwargs = Logger.default_wandb_kwargs('test_project', mode='offline', dir=str(tmpdir))
    logger = Logger('wandb_video', results_dir=tmpdir, fps=30, wandb_kwargs=wandb_kwargs)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    logger.record_frame(frame)
    logger.stop_recording()

    assert len(logger.recorded_videos) == 1
    assert logger.recorded_videos[0].exists()

    logger.log_video(0)

    logger.finish()


def test_log_hyperparameters(tmpdir):
    logger = Logger('test_params', results_dir=tmpdir)

    logger.log_hyperparameters(gamma=0.99, horizon=200, name='experiment')

    with open(tmpdir / 'test_params' / 'params.json') as params_file:
        params = json.load(params_file)

    assert params == dict(gamma=0.99, horizon=200, name='experiment')


def test_log_hyperparameters_seed(tmpdir):
    logger = Logger('test_params', results_dir=tmpdir, seed=42)

    logger.log_hyperparameters(gamma=0.99)

    assert (tmpdir / 'test_params' / 'params-42.json').exists()
    assert not (tmpdir / 'test_params' / 'params.json').exists()


def test_log_hyperparameters_no_results_dir():
    logger = Logger('test_log_hyperparameters_no_results_dir', results_dir=None)

    logger.log_hyperparameters(gamma=0.99)


def test_log_hyperparameters_non_serializable(tmpdir):
    logger = Logger('test_log_hyperparameters_non_serializable', results_dir=tmpdir)

    logger.log_hyperparameters(optimizer=optim.Adam)

    with open(tmpdir / 'test_log_hyperparameters_non_serializable' / 'params.json') as params_file:
        params = json.load(params_file)

    assert params['optimizer'] == str(optim.Adam)


def test_log_experiment_info(tmpdir):
    logger = Logger('test_info', results_dir=tmpdir)
    mdp = SimpleChain(n_states=5, goal_states=[2])

    logger.log_experiment_info(QLearning, mdp, gamma=0.9, n_steps=100)

    with open(tmpdir / 'test_info' / 'params.json') as params_file:
        params = json.load(params_file)

    assert params == dict(gamma=0.9, n_steps=100)


def test_log_experiment_info_instance(tmpdir):
    logger = Logger('test_log_experiment_info_instance', results_dir=tmpdir)
    mdp = SimpleChain(n_states=5, goal_states=[2])
    agent = QLearning(mdp.info, EpsGreedy(epsilon=Parameter(value=.1)), Parameter(value=.1))

    logger.log_experiment_info(agent, mdp)

    with open(tmpdir / 'test_log_experiment_info_instance' / 'params.json') as params_file:
        params = json.load(params_file)

    assert params == dict()


def test_log_experiment_info_no_mdp(tmpdir):
    logger = Logger('test_log_experiment_info_no_mdp', results_dir=tmpdir)

    logger.log_experiment_info(QLearning, n_epochs=3)

    with open(tmpdir / 'test_log_experiment_info_no_mdp' / 'params.json') as params_file:
        params = json.load(params_file)

    assert params == dict(n_epochs=3)
