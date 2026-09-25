from mushroom_rl.core.vectorized_env import VectorizedEnvironment


class Core(object):
    """
    Implements the functions to run a generic algorithm.

    This is a facade that, depending on the environment provided, dispatches to a single-environment
    (:class:`SequentialCore`) or a vectorized (:class:`VectorizedCore`) implementation.
    Both expose the same interface, so user code only ever instantiates ``Core``.

    """
    def __new__(cls, agent, env, *args, **kwargs):
        if cls is not Core:
            return super().__new__(cls)
        if isinstance(env, VectorizedEnvironment) and env.number > 1:
            return super().__new__(VectorizedCore)
        return super().__new__(SequentialCore)

    def __init__(self, agent, env, callbacks_fit=None, callback_step=None, logger=None):
        """
        Constructor.

        Args:
            agent (Agent): the agent moving according to a policy;
            env (Environment): the environment in which the agent moves;
            callbacks_fit (list): list of callbacks to execute at the end of each fit. The dataset view they receive
                is only valid for the duration of the callback;
            callback_step (Callback): callback to execute after each step;
            logger (Logger, None): the logger to be used by the agent. If provided, it is set on the agent via
                ``agent.set_logger`` and the video fps is configured from the environment.

        """
        self.agent = agent
        self.env = env
        self.callbacks_fit = callbacks_fit if callbacks_fit is not None else list()
        self.callback_step = callback_step if callback_step is not None else lambda x: None

        self._state = None
        self._policy_state = None
        self._current_theta = None
        self._episode_steps = None

        self._core_logic = self._create_core_logic()

        if logger is not None:
            self.set_logger(logger)

    def learn(self, n_steps=None, n_episodes=None, n_steps_per_fit=None, n_episodes_per_fit=None,
              render=False, record=False, quiet=False):
        """
        This function moves the agent in the environment and fits the policy using the collected samples.
        The agent can be moved for a given number of steps or a given number of episodes and, independently of this
        choice, the policy can be fitted after a given number of steps or a given number of episodes.
        The environment is reset at the beginning of the learning process. If ``n_steps``/``n_episodes`` is not an
        exact multiple of ``n_steps_per_fit``/``n_episodes_per_fit``, the trailing samples collected after the last
        fit are discarded rather than triggering a final, undersized fit.

        Args:
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            n_steps_per_fit (int, None): number of steps between each fit of the
                policy. With a vectorized environment it cannot be lower than the number of environments;
            n_episodes_per_fit (int, None): number of episodes between each fit
                of the policy;
            render (bool, False): whether to render the environment or not;
            record (bool, False): whether to record a video of the environment or not. If True, also the render flag
                should be set to True.
            quiet (bool, False): whether to show the progress bar or not.

        """
        assert (render and record) or (not record), "To record, the render flag must be set to true"
        if record:
            assert self.agent.logger is not None, "To record, a logger must be set via Core.set_logger"

        self._core_logic.initialize_learn(n_steps_per_fit, n_episodes_per_fit)

        dataset = self._prepare_dataset(n_steps_per_fit, n_episodes_per_fit, n_episodes is not None)

        self._run(dataset, n_steps, n_episodes, render, quiet, record)

    def evaluate(self, initial_states=None, n_steps=None, n_episodes=None, render=False, quiet=False, record=False,
                 greedy=False):
        """
        This function moves the agent in the environment using its policy.
        The agent is moved for a provided number of steps, episodes, or from a set of initial states for the whole
        episode. The environment is reset at the beginning of the learning process.

        Args:
            initial_states (Array, None): the starting states of each episode;
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            render (bool, False): whether to render the environment or not;
            quiet (bool, False): whether to show the progress bar or not;
            record (bool, False): whether to record a video of the environment or not. If True, also the render flag
                should be set to True;
            greedy (bool, False): whether the agent acts greedily, using the mode of its policy instead of
                sampling. Requires the policy to define a greedy action.

        Returns:
            The collected dataset.

        """
        assert (render and record) or (not record), "To record, the render flag must be set to true"
        if record:
            assert self.agent.logger is not None, "To record, a logger must be set via Core.set_logger"

        self._core_logic.initialize_evaluate()

        n_episodes_dataset = len(initial_states) if initial_states is not None else n_episodes
        dataset = self._prepare_dataset(n_steps, n_episodes_dataset, n_episodes is not None)

        return self._run(dataset, n_steps, n_episodes, render, quiet, record, initial_states, greedy)

    def set_logger(self, logger):
        """
        Set the logger on the agent and configure the video fps from the environment.

        Args:
            logger (Logger): the logger to be used by the agent.

        """
        self.agent.set_logger(logger)
        logger.set_video_fps(int(1 / self.env.info.dt))

    def _create_core_logic(self):
        """
        Returns:
            The :class:`CoreLogic` instance driving the step/episode counters of this core.

        """
        raise NotImplementedError

    def _prepare_dataset(self, n_steps, n_episodes, core_counts_episodes):
        """
        Build the empty dataset used to collect the samples of a run.

        Args:
            n_steps (int, None): number of steps used to size the dataset;
            n_episodes (int, None): number of episodes used to size the dataset;
            core_counts_episodes (bool): whether the run is driven by an episode count.

        Returns:
            The empty dataset to be filled during the run.

        """
        raise NotImplementedError

    def _run(self, dataset, n_steps, n_episodes, render, quiet, record, initial_states=None, greedy=False):
        raise NotImplementedError

    def _preprocess(self, state):
        """
        Method to apply state preprocessors.

        Args:
            state (Array): the state to be preprocessed.

        Returns:
             The preprocessed state.

        """
        for p in self.agent.core_preprocessors:
            p.update(state)
            state = p(state)

        return state


from ._impl.core import SequentialCore, VectorizedCore  # noqa: E402
