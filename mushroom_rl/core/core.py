from mushroom_rl.core.dataset import Dataset
from mushroom_rl.utils.record import VideoRecorder

from ._impl import CoreLogic


class Core(object):
    """
    Implements the functions to run a generic algorithm.

    """
    def __init__(self, agent, env, callbacks_fit=None, callback_step=None, record_dictionary=None):
        """
        Constructor.

        Args:
            agent (Agent): the agent moving according to a policy;
            env (Environment): the environment in which the agent moves;
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

        self._core_logic = CoreLogic()

        if record_dictionary is None:
            record_dictionary = dict()
        self._record = self._build_recorder_class(**record_dictionary)

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

        dataset = Dataset(self.env.info, self.agent.info, n_steps_per_fit, n_episodes_per_fit)

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

        self._core_logic.initialize_evaluate()

        n_episodes_dataset = len(initial_states) if initial_states is not None else n_episodes
        dataset = Dataset(self.env.info, self.agent.info, n_steps, n_episodes_dataset)

        return self._run(dataset, n_steps, n_episodes, render, quiet, record, initial_states)

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
            self._core_logic.after_step(sample[5])

            dataset.append(sample, step_info)

            if self._core_logic.fit_required():
                self.agent.fit(dataset)
                self._core_logic.after_fit()

                for c in self.callbacks_fit:
                    c(dataset)

                dataset.clear()

            last = sample[5]

        self.agent.stop()
        self.env.stop()

        self._end(record)

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
                self._record(frame)

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
        self._policy_state, self._current_theta = self.agent.episode_start(state, episode_info)
        self._state = self._preprocess(state)
        self.agent.next_action = None

        self._episode_steps = 0

    def _end(self, record):
        self._state = None
        self._policy_state = None
        self._current_theta = None
        self._episode_steps = None

        if record:
            self._record.stop()

        self._core_logic.terminate_run()

    def _preprocess(self, state):
        """
        Method to apply state preprocessors.

        Args:
            state (np.ndarray): the state to be preprocessed.

        Returns:
             The preprocessed state.

        """
        for p in self.agent.preprocessors:
            p.update(state)
            state = p(state)

        return state

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
