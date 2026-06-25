import pickle

try:
    import wandb
except ImportError:
    wandb = None


class WandbLogger(object):
    """
    This class implements the wandb logging functionality. It is enabled only if
    the ``wandb`` package is installed and a set of init arguments is provided,
    otherwise every method is a no-op.

    """
    def __init__(self, wandb_kwargs=None, results_dir=None, log_dir=None, append=False):
        """
        Constructor.

        Args:
            wandb_kwargs (dict, None): dictionary of arguments forwarded to
                ``wandb.init``. If None, or if the ``wandb`` package is not
                installed, wandb logging is disabled and all methods are no-ops;
            results_dir (Path, None): logging directory created by the
                ``Logger``. If provided, and ``dir`` is not already set in
                ``wandb_kwargs``, wandb stores its files inside this directory;
            log_dir (Path, None): experiment-specific directory where the wandb
                run state is persisted for resume on append;
            append (bool, False): if True and a previous run state is found in
                ``log_dir``, the wandb run is resumed with the same run id and
                the fit counter is restored.

        """
        self._wandb_run = None
        self._n_fit = 0
        self._log_dir = log_dir

        if wandb_kwargs is not None and wandb is not None:
            if results_dir is not None and 'dir' not in wandb_kwargs:
                wandb_kwargs = dict(wandb_kwargs, dir=str(results_dir))

            if append:
                state = self._load_wandb_state()
                if state is not None and 'resume' not in wandb_kwargs:
                    wandb_kwargs = dict(wandb_kwargs, resume='allow', id=state['run_id'])
                    self._n_fit = state.get('n_fit', 0)

            self._wandb_run = wandb.init(**wandb_kwargs)

            wandb.define_metric('n_fit')
            wandb.define_metric('training/*', step_metric='n_fit')
            wandb.define_metric('epoch')
            wandb.define_metric('eval/*', step_metric='epoch')

            self._save_wandb_state()

    @staticmethod
    def default_wandb_kwargs(project, config=None, **overrides):
        """
        Build a default dictionary of arguments for ``wandb.init``. The returned
        dictionary can be freely edited and is meant to be passed to the
        ``Logger`` constructor through the ``wandb_kwargs`` argument.

        Args:
            project (str): name of the wandb project;
            config (dict, None): dictionary of hyperparameters to log;
            **overrides: any additional key overrides the defaults.

        Returns:
            The dictionary of arguments for ``wandb.init``.

        """
        kwargs = dict(
            project=project,
            entity=None,
            group=None,
            name=None,
            tags=None,
            config=config if config is not None else dict(),
            mode='online',
        )
        kwargs.update(overrides)

        return kwargs

    def log_wandb_training(self, **kwargs):
        """
        Log a set of named training metrics to wandb, grouped under the ``training/`` prefix
        and using the number of fits as x-axis.

        Args:
            **kwargs: set of named values to be logged.

        """
        if self._wandb_run is not None:
            data = {'training/' + name: value for name, value in kwargs.items()}
            data['n_fit'] = self._n_fit
            wandb.log(data)

    def log_wandb_eval(self, epoch, **kwargs):
        """
        Log a set of named evaluation metrics to wandb, grouped under the ``eval/`` prefix
        and using the epoch as x-axis.

        Args:
            epoch (int): the current epoch, used as x-axis for the logged values;
            **kwargs: set of named values to be logged.

        """
        if self._wandb_run is not None:
            data = {'eval/' + name: value for name, value in kwargs.items()}
            data['epoch'] = epoch
            wandb.log(data)

    def log_wandb_video(self, name, path, epoch):
        """
        Log a video file to wandb, grouped under the ``eval/`` prefix and using the epoch as
        x-axis. The video is uploaded as is, without any re-encoding.

        Args:
            name (str): the name of the video;
            path (str, Path): the path to the video file to upload;
            epoch (int): the current epoch, used as x-axis for the video.

        """
        if self._wandb_run is not None:
            video = wandb.Video(str(path))
            wandb.log({'eval/' + name: video, 'epoch': epoch})

    def advance_step(self):
        """
        Advance the number of fits counter by one. To be called once per fit, so that all
        the training values logged during a fit share the same ``n_fit`` x-axis value.

        """
        if self._wandb_run is not None:
            self._n_fit += 1

    def finish(self):
        """
        Finish the current wandb run, flushing the data to disk. No-op if wandb logging
        is not active.

        """
        if self._wandb_run is not None:
            self._save_wandb_state()
            self._wandb_run.finish()
            self._wandb_run = None

    def _save_wandb_state(self):
        if self._log_dir is not None and self._wandb_run is not None:
            state = {'run_id': self._wandb_run.id, 'n_fit': self._n_fit}
            with open(self._log_dir / '.wandb_state.pkl', 'wb') as f:
                pickle.dump(state, f)

    def _load_wandb_state(self):
        if self._log_dir is not None:
            state_file = self._log_dir / '.wandb_state.pkl'
            if state_file.exists():
                with open(state_file, 'rb') as f:
                    return pickle.load(f)
        return None

    @property
    def wandb_active(self):
        """
        Returns:
            True if wandb logging is enabled, False otherwise.

        """
        return self._wandb_run is not None
