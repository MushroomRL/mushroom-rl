from mushroom_rl.core.core import Core
from mushroom_rl.core.dataset import Dataset

from .core_logic import CoreLogic


class SequentialCore(Core):
    """
    Single-environment implementation of :class:`Core`.

    """
    def _create_core_logic(self):
        return CoreLogic()

    def _prepare_dataset(self, n_steps, n_episodes, core_counts_episodes):
        return Dataset.generate(self.env.info, self.agent.info, n_steps, n_episodes,
                                core_counts_episodes=core_counts_episodes)

    def _run(self, dataset, n_steps, n_episodes, render, quiet, record, initial_states=None, greedy=False):
        self._core_logic.initialize_run(n_steps, n_episodes, initial_states, quiet)

        draw_action = self.agent.draw_action_greedy if greedy else self.agent.draw_action

        last = True
        while self._core_logic.move_required():
            if last:
                self._reset(dataset, initial_states, greedy)
                if self.agent.info.is_episodic:
                    dataset.append_theta(self._current_theta)

            sample, step_info = self._step(draw_action, render, record)

            self.callback_step(sample)
            last = self._core_logic.after_step(sample[5])

            dataset.append(sample, step_info)

            if self._core_logic.fit_required():
                self.agent.fit(dataset)
                self._core_logic.after_fit()

                for c in self.callbacks_fit:
                    c(dataset)

                dataset.clear(self.agent.history_manager.history_context())

        self.agent.stop()
        self.env.stop()

        self._end(record)

        return dataset

    def _step(self, draw_action, render, record):
        """
        Single step.

        Args:
            draw_action (callable): the agent method used to draw the action (stochastic or greedy);
            render (bool): whether to render or not.

        Returns:
            A tuple containing the previous state, the action sampled by the agent, the reward obtained, the reached
            state, the absorbing flag of the reached state and the last step flag.

        """
        action = draw_action(self._state)
        next_state, reward, absorbing, step_info = self.env.step(action)

        if render:
            frame = self.env.render(record)

            if record:
                self.agent.logger.record_frame(frame)

        self._episode_steps += 1

        last = self._episode_steps >= self.env.info.horizon or absorbing

        state = self._state
        next_state = self._preprocess(next_state)
        self._state = next_state

        policy_state = self._policy_state
        policy_next_state = self.agent.policy_state
        self._policy_state = policy_next_state

        return (state, action, reward, next_state, absorbing, last, policy_state, policy_next_state), step_info

    def _reset(self, dataset, initial_states, greedy=False):
        """
        Reset the state of the agent and store the information the environment reports.

        Args:
            dataset (Dataset): the dataset the episode information is appended to;
            initial_states (Array, None): the states the episodes are started from;
            greedy (bool, False): whether the agent acts greedily.

        """
        initial_state = self._core_logic.get_initial_state(initial_states)

        state, episode_info = self.env.reset(initial_state)
        dataset.append_episode_info(episode_info)
        self._state = self._preprocess(state)
        self._policy_state, self._current_theta = self.agent.episode_start(self._state, episode_info, greedy)

        self._episode_steps = 0

    def _end(self, record):
        self._state = None
        self._policy_state = None
        self._current_theta = None
        self._episode_steps = None

        if record:
            self.agent.logger.stop_recording()

        self._core_logic.terminate_run()
