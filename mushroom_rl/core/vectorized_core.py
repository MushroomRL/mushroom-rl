from mushroom_rl.core.dataset import VectorizedDataset
from mushroom_rl.utils.record import VideoRecorder

from ._impl import VectorizedCoreLogic


class VectorCore(object):
    """
    Implements the functions to run a generic algorithm.

    """

    def __init__(self, agent, env, callbacks_fit=None, callback_step=None, record_dictionary=None):
        """
        Constructor.

        Args:
            agent (Agent): the agent moving according to a policy;
            env (VectorEnvironment): the environment in which the agent moves;
            callbacks_fit (list): list of callbacks to execute at the end of each fit;
            callback_step (Callback): callback to execute after each step;
            record_dictionary (dict, None): a dictionary of parameters for the recording, must containt the
                recorder_class, fps,  and optionally other keyword arguments to be passed to build the recorder class.
                By default, the VideoRecorder class is used and the environment action frequency as frames per second.

        """
        self.agent = agent
        self.env = env
        self.callbacks_fit = callbacks_fit if callbacks_fit is not None else list()
        self.callback_step = callback_step if callback_step is not None else lambda x: None

        self._state = None
        self._policy_state = None
        self._current_theta = None
        self._episode_steps = None

        self._core_logic = VectorizedCoreLogic(self.env.info.backend, self.env.number)

        if record_dictionary is None:
            record_dictionary = dict()
        self._record = [self._build_recorder_class(**record_dictionary) for _ in range(self.env.number)]

    def learn(self, n_steps=None, n_episodes=None, n_steps_per_fit=None, n_episodes_per_fit=None,
              render=False, record=False, quiet=False):
        """
        This function moves the agent in the environment and fits the policy using the collected samples.
        The agent can be moved for a given number of steps or a given number of episodes and, independently from this
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
        self._core_logic.initialize_learn(n_steps_per_fit, n_episodes_per_fit)

        dataset = VectorizedDataset(self.env.info, self.agent.info, self.env.number,
                                    n_steps_per_fit, n_episodes_per_fit)

        self._run(dataset, n_steps, n_episodes, render, quiet, record)

    def evaluate(self, initial_states=None, n_steps=None, n_episodes=None, render=False, quiet=False, record=False):
        """
        This function moves the agent in the environment using its policy.
        The agent is moved for a provided number of steps, episodes, or from a set of initial states for the whole
        episode. The environment is reset at the beginning of the learning process.

        Args:
            initial_states (array, None): the starting states of each episode;
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

        self._core_logic.initialize_evaluate()

        n_episodes_dataset = len(initial_states) if initial_states is not None else n_episodes
        dataset = VectorizedDataset(self.env.info, self.agent.info, self.env.number, n_steps, n_episodes_dataset)

        return self._run(dataset, n_steps, n_episodes, render, quiet, record, initial_states)

    def _run(self, dataset, n_steps, n_episodes, render, quiet, record, initial_states=None):
        self._core_logic.initialize_run(n_steps, n_episodes, initial_states, quiet)

        last = self._core_logic.converter.ones(self.env.number, dtype=bool)
        mask = None

        while self._core_logic.move_required():
            if last.any():
                mask = self._core_logic.get_mask(last)
                reset_mask = self._reset(initial_states, last, mask)

                if self.agent.info.is_episodic and reset_mask.any():
                    dataset.append_theta_vectorized(self._current_theta, reset_mask)

            samples, step_infos = self._step(render, record, mask)

            self.callback_step(samples)
            self._core_logic.after_step(samples[5] & mask)

            dataset.append_vectorized(samples, step_infos, mask)

            last = samples[5]

            if self._core_logic.fit_required():
                fit_dataset = dataset.flatten()
                self.agent.fit(fit_dataset)
                last = self._core_logic.after_fit_vectorized(last)

                for c in self.callbacks_fit:
                    c(dataset)

                dataset.clear()

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
            frames = self.env.render_all(mask, record=record)

            if record:
                for i in range(self.env.number):
                    if mask[i]:
                        self._record[i](frames[i])

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
        initial_state = self._core_logic.get_initial_state(initial_states)

        state, episode_info = self._preprocess(self.env.reset_all(reset_mask, initial_state))
        self._policy_state, self._current_theta = self.agent.episode_start_vectorized(state, episode_info,
                                                                                      self.env.number)
        self._state = self._preprocess(state)
        self.agent.next_action = None

        if self._episode_steps is None:
            self._episode_steps = self._core_logic.converter.zeros(self.env.number, dtype=int)
        else:
            self._episode_steps[last] = 0

        return reset_mask

    def _end(self, record):
        self._state = None
        self._policy_state = None
        self._current_theta = None
        self._episode_steps = None

        if record:
            for record in self._record:
                record.stop()

        self._core_logic.terminate_run()

    def _preprocess(self, states):
        """
        Method to apply state preprocessors.

        Args:
            states (array): the states to be preprocessed.

        Returns:
             The preprocessed states.

        """
        for p in self.agent.preprocessors:
            states = p(states)

        return states

    def _build_recorder_class(self, recorder_class=None, fps=None, **kwargs):
        """
        Method to create a video recorder class.

        Args:
            recorder_class (class): the class used to record the video. By default, we use the ``VideoRecorder`` class
                from mushroom. The class must implement the ``__call__`` and ``stop`` methods.

        Returns:
             The recorder object.

        """

        if not recorder_class:
            recorder_class = VideoRecorder

        if not fps:
            fps = int(1 / self.env.info.dt)

        return recorder_class(fps=fps, **kwargs)
