import numpy as np

from .core_logic import CoreLogic


class VectorizedCoreLogic(CoreLogic):
    def __init__(self, n_envs):
        self._n_envs = n_envs

        super().__init__()

    def get_action_mask(self):
        action_mask = np.ones(self._n_envs, dtype=bool)

        if self._n_episodes is not None:
            if self._n_episodes_per_fit is not None:
                action_mask = self._current_episodes_counter != self._n_episodes_per_fit
            else:
                action_mask = self._current_episodes_counter != self._n_episodes

        return action_mask

    def get_initial_state(self, initial_states):

        if initial_states is None or np.all(self._total_episodes_counter == self._n_episodes):
            initial_state = None
        else:
            initial_state = initial_states[self._total_episodes_counter]  # FIXME

        return initial_state

    def after_step(self, last):
        self._total_steps_counter += self._n_envs
        self._current_steps_counter += self._n_envs
        self._steps_progress_bar.update(self._n_envs)

        completed = last.sum()
        self._total_episodes_counter += completed
        self._current_episodes_counter += completed
        self._episodes_progress_bar.update(completed)

    def after_fit(self):
        self._current_episodes_counter = np.zeros(self._n_envs, dtype=int)
        self._current_steps_counter = 0

    def _reset_counters(self):
        self._total_episodes_counter = np.zeros(self._n_envs, dtype=int)
        self._current_episodes_counter = np.zeros(self._n_envs, dtype=int)
        self._total_steps_counter = 0
        self._current_steps_counter = 0

    def _move_episodes_condition(self):
        return np.sum(self._total_episodes_counter) < self._n_episodes

    def _fit_episodes_condition(self):
        return np.sum(self._current_episodes_counter) >= self._n_episodes_per_fit
