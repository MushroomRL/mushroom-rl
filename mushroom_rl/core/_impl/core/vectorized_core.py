from mushroom_rl.core.core import Core
from mushroom_rl.core.vectorized_dataset import VectorizedDataset

from .vectorized_core_logic import VectorizedCoreLogic


class VectorizedCore(Core):
    """
    Vectorized (multienvironment) implementation of :class:`Core`.

    """
    def _create_core_logic(self):
        return VectorizedCoreLogic(self.env.info.backend, self.env.number, self.env.info.device)

    def _prepare_dataset(self, n_steps, n_episodes, core_counts_episodes):
        return VectorizedDataset.generate(self.env.info, self.agent.info, n_steps, n_episodes,
                                          self.env.number, core_counts_episodes)

    def _run(self, dataset, n_steps, n_episodes, render, quiet, record, initial_states=None, greedy=False):
        self._core_logic.initialize_run(n_steps, n_episodes, initial_states, quiet)

        draw_action = self.agent.draw_action_greedy if greedy else self.agent.draw_action

        last = self._core_logic.converter.ones(self.env.number, dtype=bool, device=self.env.info.device)
        need_reset = True

        while self._core_logic.move_required():
            mask = self._core_logic.get_mask(last)

            if need_reset:
                current_theta, reset_mask = self._reset(dataset, initial_states, last, mask, greedy)

                if self.agent.info.is_episodic and reset_mask.any():
                    dataset.append_theta_vectorized(current_theta, reset_mask)

            samples, step_infos = self._step(draw_action, render, record, mask)

            self.callback_step(samples)
            completed = self._core_logic.after_step(samples[5] & mask)

            dataset.append_vectorized(samples, step_infos, mask)

            last = samples[5]
            need_reset = completed > 0

            if self._core_logic.fit_required():
                consumed = dataset.consume(self._core_logic.n_steps_per_fit)
                self.agent.fit(consumed.flatten())

                for c in self.callbacks_fit:
                    c(consumed)

                n_carry_forward_steps = dataset.clear(keep_leftovers=True,
                                                      history_context=self.agent.history_manager.history_context())
                last = self._core_logic.after_fit_vectorized(last, n_carry_forward_steps)
                if self._core_logic.n_episodes_per_fit is not None:
                    need_reset = True

        self.agent.stop()
        self.env.stop()

        self._end(record)

        return dataset.flatten()

    def _step(self, draw_action, render, record, mask):
        """
        Single step.

        Args:
            draw_action (callable): the agent method used to draw the action (stochastic or greedy);
            render (bool): whether to render or not.

        Returns:
            A tuple containing the previous states, the actions sampled by the
            agent, the rewards obtained, the reached states, the absorbing flags
            of the reached states and the last step flags.

        """
        action = draw_action(self._state)

        next_state, rewards, absorbing, step_info = self.env.step_all(mask, action)

        self._episode_steps[mask] += 1

        if render:
            frame = self.env.render_all(mask, record=record)

            if record:
                self.agent.logger.record_frame(frame, mask)

        last = absorbing | (self._episode_steps >= self.env.info.horizon)

        state = self._state
        next_state = self._preprocess_masked(next_state, mask, self._core_logic.n_active_envs)
        self._state = next_state

        policy_state = self._policy_state
        policy_next_state = self.agent.policy_state
        self._policy_state = policy_next_state

        return (state, action, rewards, next_state, absorbing, last, policy_state, policy_next_state), step_info

    def _reset(self, dataset, initial_states, last, mask, greedy=False):
        """
        Reset the states of the agent and store the information the environments that reset report.

        Args:
            dataset (VectorizedDataset): the dataset the episode information is appended to;
            initial_states (Array, None): the states the episodes are started from;
            last (Array): boolean mask marking the environments whose episode ended;
            mask (Array): boolean mask marking the active environments;
            greedy (bool, False): whether the agent acts greedily.

        """
        reset_mask = last & mask

        initial_state = self._core_logic.get_initial_state(initial_states, reset_mask)

        state, episode_info = self.env.reset_all(reset_mask, initial_state)
        dataset.append_episode_info(episode_info, reset_mask)

        self._state = self._preprocess_masked(state, reset_mask, self._core_logic.n_reset_envs)

        policy_state, current_theta = self.agent.episode_start_vectorized(self._state, episode_info, reset_mask,
                                                                          greedy)
        self._policy_state = policy_state

        if self._episode_steps is None:
            self._episode_steps = self._core_logic.converter.zeros(self.env.number, dtype=int,
                                                                   device=self.env.info.device)
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

    def _preprocess_masked(self, state, mask, n_selected):
        """
        Apply the state preprocessors to the observations of the environments selected by the mask, leaving
        the observations of the other environments unprocessed.

        Args:
            state (Array): the observations of every environment;
            mask (Array): mask selecting the environments whose observations must be preprocessed;
            n_selected (int): the number of environments selected by the mask.

        Returns:
            The state of every environment, in a new array.

        """
        if n_selected == self.env.number:
            return self._preprocess(state)

        carried_state = state if self._state is None else self._state

        if n_selected > 0:
            converter = self._core_logic.converter
            selected_state = converter.masked_select(state, mask)
            preprocessed_state = self._preprocess(selected_state)
            carried_state = converter.copy(carried_state)
            converter.masked_assign(carried_state, mask, preprocessed_state)

        return carried_state
