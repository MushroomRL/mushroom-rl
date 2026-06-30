from mushroom_rl.core.dataset import Dataset, VectorizedDataset
from mushroom_rl.core.vectorized_env import VectorizedEnvironment

from ._impl import CoreLogic, VectorizedCoreLogic


class Core(object):
    """
    Implements the functions to run a generic algorithm.

    This is a facade that, depending on the type of environment provided, dispatches to a
    single-environment (:class:`SequentialCore`) or a vectorized (:class:`VectorizedCore`)
    implementation. Both expose the same interface, so user code only ever instantiates ``Core``.

    """
    def __new__(cls, agent, env, *args, **kwargs):
        if cls is not Core:
            return super().__new__(cls)
        if isinstance(env, VectorizedEnvironment):
            return super().__new__(VectorizedCore)
        return super().__new__(SequentialCore)

    def __init__(self, agent, env, callbacks_fit=None, callback_step=None, logger=None):
        """
        Constructor.

        Args:
            agent (Agent): the agent moving according to a policy;
            env (Environment): the environment in which the agent moves;
            callbacks_fit (list): list of callbacks to execute at the end of each fit;
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
        The environment is reset at the beginning of the learning process.

        Args:
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            n_steps_per_fit (int, None): number of steps between each fit of the
                policy;
            n_episodes_per_fit (int, None): number of episodes between each fit
                of the policy;
            render (bool, False): whether to render the environment or not;
            record (bool, False): whether to record a video of the environment or not. If True, also the render flag
                should be set to True.
            quiet (bool, False): whether to show the progress bar or not.

        """
        assert (render and record) or (not record), "To record, the render flag must be set to true"
        if record:
            assert self.agent.logger is not None, "To record, a logger must be set on the agent via set_logger"

        self._core_logic.initialize_learn(n_steps_per_fit, n_episodes_per_fit)

        dataset = self._prepare_dataset(n_steps_per_fit, n_episodes_per_fit, n_episodes is not None)

        self._run(dataset, n_steps, n_episodes, render, quiet, record)

    def evaluate(self, initial_states=None, n_steps=None, n_episodes=None, render=False, quiet=False, record=False):
        """
        This function moves the agent in the environment using its policy.
        The agent is moved for a provided number of steps, episodes, or from a set of initial states for the whole
        episode. The environment is reset at the beginning of the learning process.

        Args:
            initial_states (np.ndarray, None): the starting states of each episode;
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            render (bool, False): whether to render the environment or not;
            quiet (bool, False): whether to show the progress bar or not;
            record (bool, False): whether to record a video of the environment or not. If True, also the render flag
                should be set to True.

        Returns:
            The collected dataset.

        """
        assert (render and record) or (not record), "To record, the render flag must be set to true"
        if record:
            assert self.agent.logger is not None, "To record, a logger must be set on the agent via set_logger"

        self._core_logic.initialize_evaluate()

        n_episodes_dataset = len(initial_states) if initial_states is not None else n_episodes
        dataset = self._prepare_dataset(n_steps, n_episodes_dataset, n_episodes is not None)

        return self._run(dataset, n_steps, n_episodes, render, quiet, record, initial_states)

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

    def _run(self, dataset, n_steps, n_episodes, render, quiet, record, initial_states=None):
        raise NotImplementedError

    def _preprocess(self, state):
        """
        Method to apply state preprocessors.

        Args:
            state (np.ndarray): the state to be preprocessed.

        Returns:
             The preprocessed state.

        """
        for p in self.agent.core_preprocessors:
            p.update(state)
            state = p(state)

        return state


class SequentialCore(Core):
    """
    Single-environment implementation of :class:`Core`.

    """
    def _create_core_logic(self):
        return CoreLogic()

    def _prepare_dataset(self, n_steps, n_episodes, core_counts_episodes):
        return Dataset.generate(self.env.info, self.agent.info, n_steps, n_episodes,
                                core_counts_episodes=core_counts_episodes)

    def _run(self, dataset, n_steps, n_episodes, render, quiet, record, initial_states=None):
        self._core_logic.initialize_run(n_steps, n_episodes, initial_states, quiet)

        last = True
        while self._core_logic.move_required():
            if last:
                self._reset(initial_states)
                if self.agent.info.is_episodic:
                    dataset.append_theta(self._current_theta)

            sample, step_info = self._step(render, record)

            self.callback_step(sample)
            last = self._core_logic.after_step(sample[5])

            dataset.append(sample, step_info)

            if self._core_logic.fit_required():
                self.agent.fit(dataset)
                self._core_logic.after_fit()

                for c in self.callbacks_fit:
                    c(dataset)

                dataset.clear()

        self.agent.stop()
        self.env.stop()

        self._end(record)

        dataset.info.parse()
        dataset.episode_info.parse()
        return dataset

    def _step(self, render, record):
        """
        Single step.

        Args:
            render (bool): whether to render or not.

        Returns:
            A tuple containing the previous state, the action sampled by the agent, the reward obtained, the reached
            state, the absorbing flag of the reached state and the last step flag.

        """
        action, policy_next_state = self.agent.draw_action(self._state, self._policy_state)
        next_state, reward, absorbing, step_info = self.env.step(action)

        if render:
            frame = self.env.render(record)

            if record:
                self.agent.logger.record_frame(frame)

        self._episode_steps += 1

        last = self._episode_steps >= self.env.info.horizon or absorbing

        state = self._state
        policy_state = self._policy_state
        next_state = self._preprocess(next_state)
        self._state = next_state
        self._policy_state = policy_next_state

        return (state, action, reward, next_state, absorbing, last, policy_state, policy_next_state), step_info

    def _reset(self, initial_states):
        """
        Reset the state of the agent.

        """
        initial_state = self._core_logic.get_initial_state(initial_states)

        state, episode_info = self.env.reset(initial_state)
        self._state = self._preprocess(state)
        self._policy_state, self._current_theta = self.agent.episode_start(self._state, episode_info)
        self.agent.next_action = None

        self._episode_steps = 0

    def _end(self, record):
        self._state = None
        self._policy_state = None
        self._current_theta = None
        self._episode_steps = None

        if record:
            self.agent.logger.stop_recording()

        self._core_logic.terminate_run()


class VectorizedCore(Core):
    """
    Vectorized (multienvironment) implementation of :class:`Core`.

    """
    def _create_core_logic(self):
        return VectorizedCoreLogic(self.env.info.backend, self.env.number)

    def _prepare_dataset(self, n_steps, n_episodes, core_counts_episodes):
        return VectorizedDataset.generate(self.env.info, self.agent.info, n_steps, n_episodes,
                                          self.env.number, core_counts_episodes)

    def _run(self, dataset, n_steps, n_episodes, render, quiet, record, initial_states=None):
        self._core_logic.initialize_run(n_steps, n_episodes, initial_states, quiet)

        last = self._core_logic.converter.ones(self.env.number, dtype=bool)
        mask = None
        need_reset = True

        while self._core_logic.move_required():
            if need_reset:
                mask = self._core_logic.get_mask(last)
                current_theta, reset_mask = self._reset(initial_states, last, mask)

                if self.agent.info.is_episodic and reset_mask.any():
                    dataset.append_theta_vectorized(current_theta, reset_mask)

            samples, step_infos = self._step(render, record, mask)

            self.callback_step(samples)
            completed = self._core_logic.after_step(samples[5] & mask)

            dataset.append_vectorized(samples, step_infos, mask)

            last = samples[5]
            need_reset = completed > 0

            if self._core_logic.fit_required():
                fit_dataset = dataset.flatten(self._core_logic.n_steps_per_fit)
                self.agent.fit(fit_dataset)

                for c in self.callbacks_fit:
                    c(dataset)

                n_carry_forward_steps = dataset.clear(self._core_logic.n_steps_per_fit)
                last = self._core_logic.after_fit_vectorized(last, n_carry_forward_steps)
                if self._core_logic.n_episodes_per_fit is not None:
                    need_reset = True

        self.agent.stop()
        self.env.stop()

        self._end(record)

        return dataset.flatten()

    def _step(self, render, record, mask):
        """
        Single step.

        Args:
            render (bool): whether to render or not.

        Returns:
            A tuple containing the previous states, the actions sampled by the
            agent, the rewards obtained, the reached states, the absorbing flags
            of the reached states and the last step flags.

        """
        action, policy_next_state = self.agent.draw_action(self._state, self._policy_state)

        next_state, rewards, absorbing, step_info = self.env.step_all(mask, action)

        self._episode_steps[mask] += 1

        if render:
            frame = self.env.render_all(mask, record=record)

            if record:
                self.agent.logger.record_frame(frame)

        last = absorbing | (self._episode_steps >= self.env.info.horizon)

        state = self._state
        policy_state = self._policy_state
        next_state = self._preprocess(next_state)
        self._state = next_state
        self._policy_state = policy_next_state

        return (state, action, rewards, next_state, absorbing, last, policy_state, policy_next_state), step_info

    def _reset(self, initial_states, last, mask):
        """
        Reset the states of the agent.

        """
        reset_mask = last & mask

        initial_state = self._core_logic.get_initial_state(initial_states, reset_mask)

        state, episode_info = self.env.reset_all(reset_mask, initial_state)

        self._state = self._preprocess(state)
        policy_state, current_theta = self.agent.episode_start_vectorized(self._state, episode_info, reset_mask)
        if self._policy_state is None or policy_state is None:
            self._policy_state = policy_state
        elif reset_mask.any():
            self._policy_state[reset_mask] = policy_state[reset_mask]
        self.agent.next_action = None

        if self._episode_steps is None:
            self._episode_steps = self._core_logic.converter.zeros(self.env.number, dtype=int)
        else:
            self._episode_steps[last] = 0

        return current_theta, reset_mask

    def _end(self, record):
        self._state = None
        self._policy_state = None
        self._episode_steps = None

        if record:
            self.agent.logger.stop_recording()

        self._core_logic.terminate_run()
