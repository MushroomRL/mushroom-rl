from datetime import datetime
from pathlib import Path

from mushroom_rl.core.logger.console_logger import ConsoleLogger
from mushroom_rl.core.logger.data_logger import DataLogger
from mushroom_rl.core.logger.video_logger import VideoLogger
from mushroom_rl.core.logger.wandb_logger import WandbLogger


class Logger(DataLogger, ConsoleLogger, VideoLogger, WandbLogger):
    """
    This class implements the logging functionality. It can be used to create
    automatically a log directory, save numpy data array and the current agent.
    It optionally logs to Weights & Biases (wandb), if the ``wandb`` package is
    installed and a set of init arguments is provided.

    """
    def __init__(self, log_name='', results_dir='./logs', log_console=False,
                 use_timestamp=False, append=False, seed=None, wandb_kwargs=None,
                 force_numpy=False, recorder_class=None, fps=None, recorder_kwargs=None,
                 **kwargs):
        """
        Constructor.

        Args:
            log_name (string, ''): name of the current experiment directory if not
                specified, the current timestamp is used.
            results_dir (string, './logs'): name of the base logging directory.
                If set to None, no directory is created;
            log_console (bool, False): whether to log or not the console output;
            use_timestamp (bool, False): If true, adds the current timestamp to
                the folder name;
            append (bool, False): If true, the logger will append the new
                data logged to the one already existing in the directory;
            seed (int, None): seed for the current run. It can be optionally
                specified to add a seed suffix for each data file logged.
                When wandb logging is active, the seed is added to the wandb
                ``config`` and, if ``name`` is not set, to the run name;
            wandb_kwargs (dict, None): dictionary of arguments forwarded to
                ``wandb.init`` to enable wandb logging. If None, or if the
                ``wandb`` package is not installed, wandb logging is disabled.
                Use ``Logger.default_wandb_kwargs`` to build a default dictionary.
                If ``group`` is not set, it defaults to ``log_name`` so that
                all runs from the same experiment are grouped together;
            force_numpy (bool, False): if True, the values logged through the
                ``log`` method are also stored on disk as numpy arrays (only if a
                results directory is set);
            recorder_class (class, None): the class used to record video. By default,
                the ``VideoRecorder`` class is used. The class must implement the
                ``__call__`` and ``stop`` methods;
            fps (int, None): frames per second for video recording. If None, the
                value is set automatically by ``Core.set_logger`` from the environment;
            recorder_kwargs (dict, None): additional keyword arguments forwarded to
                the recorder class constructor;
            **kwargs: other parameters for ConsoleLogger class.

        """

        if log_console:
            assert results_dir is not None

        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        if not log_name:
            log_name = timestamp
        elif use_timestamp:
            log_name += '_' + timestamp

        base_results_dir = Path(results_dir) if results_dir else None

        if results_dir:
            results_dir = base_results_dir / log_name
            results_dir.mkdir(parents=True, exist_ok=True)

        suffix = '' if seed is None else '-' + str(seed)

        self._force_numpy = force_numpy and results_dir is not None

        video_path = results_dir / 'videos' if results_dir else None

        DataLogger.__init__(self, results_dir, suffix=suffix, append=append)
        ConsoleLogger.__init__(self, log_name, results_dir if log_console else None,
                               suffix=suffix, **kwargs)
        VideoLogger.__init__(self, recorder_class=recorder_class, fps=fps,
                             video_path=video_path, append=append, **(recorder_kwargs or {}))
        if wandb_kwargs is not None:
            if not wandb_kwargs.get('group'):
                wandb_kwargs = dict(wandb_kwargs, group=log_name)

            if seed is not None:
                config = dict(wandb_kwargs.get('config') or {})
                config['seed'] = seed
                wandb_kwargs = dict(wandb_kwargs, config=config)

                if not wandb_kwargs.get('name'):
                    wandb_kwargs = dict(wandb_kwargs, name=log_name + '_' + str(seed))

        WandbLogger.__init__(self, wandb_kwargs, base_results_dir, log_dir=results_dir, append=append)

    def log_training(self, **kwargs):
        """
        Log a set of named training metrics. The values are logged to wandb under the
        ``training/`` group (if active), to the console with the ``debug`` level (so they
        are not shown by default), and to disk as numpy arrays inside the ``training``
        subfolder only if the logger was constructed with ``force_numpy=True``.

        A ``'/'`` in a name groups the metric in wandb (e.g. ``'critic/loss'``) and is
        replaced by ``'_'`` for the numpy file name (e.g. ``critic_loss.npy``).

        Args:
            **kwargs: set of named values to be logged.

        """
        self.log_wandb_training(**kwargs)

        if self._force_numpy:
            numpy_kwargs = {name.replace('/', '_'): data for name, data in kwargs.items()}
            self.log_numpy(folder='training', **numpy_kwargs)

        self.debug(' '.join(f'{name}: {data}' for name, data in kwargs.items()))

    def log_evaluation(self, epoch, **kwargs):
        """
        Log a set of named evaluation metrics. The values are logged to wandb under the
        ``eval/`` group using the epoch as x-axis (if active), to the console through
        ``epoch_info``, and to disk as numpy arrays in the logging directory.

        Args:
            epoch (int): the current epoch;
            **kwargs: set of named values to be logged.

        """
        self.log_wandb_eval(epoch, **kwargs)

        if self._results_dir is not None:
            numpy_kwargs = {name.replace('/', '_'): data for name, data in kwargs.items()}
            self.log_numpy(**numpy_kwargs)

        self.epoch_info(epoch, **kwargs)

    def log_video(self, epoch, video=None, wandb_name='evaluation'):
        """
        If wandb logging is active, upload a video to wandb under the ``video/`` group using the
        epoch as x-axis. The recording itself is stopped by ``Core``; this method only handles
        the wandb upload. The video is uploaded as is, without any re-encoding. The same
        ``wandb_name`` should be used across epochs so that wandb shows a slider to browse them.

        Args:
            epoch (int): the current epoch, used as x-axis for the video;
            video (str, Path, None): path of the video file to upload. If None, the last recorded
                video is used;
            wandb_name (str, 'evaluation'): the wandb key name for the video. Must be consistent
                across epochs for the slider to work.

        """
        if not self.wandb_active:
            return

        if video is None:
            if not self._recorded_videos:
                return
            video = self._recorded_videos[-1]

        self.log_wandb_video(wandb_name, video, epoch)
