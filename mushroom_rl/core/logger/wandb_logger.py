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
    def __init__(self, wandb_kwargs=None):
        """
        Constructor.

        Args:
            wandb_kwargs (dict, None): dictionary of arguments forwarded to
                ``wandb.init``. If None, or if the ``wandb`` package is not
                installed, wandb logging is disabled and all methods are no-ops.

        """
        self._wandb_run = None
        self._wandb_step = 0

        if wandb_kwargs is not None and wandb is not None:
            self._wandb_run = wandb.init(**wandb_kwargs)

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

    def log_wandb(self, **kwargs):
        """
        Log a set of named scalars to wandb at the current step.

        Args:
            **kwargs: set of named values to be logged.

        """
        if self._wandb_run is not None:
            wandb.log(kwargs, step=self._wandb_step)

    def advance_step(self):
        """
        Advance the internal wandb step counter by one. To be called once a
        logical logging step is complete, so that all the values logged in
        between share the same wandb step.

        """
        if self._wandb_run is not None:
            self._wandb_step += 1

    @property
    def wandb_active(self):
        """
        Returns:
            True if wandb logging is enabled, False otherwise.

        """
        return self._wandb_run is not None
