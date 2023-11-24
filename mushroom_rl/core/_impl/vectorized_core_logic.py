from .type_conversions import DataConversion
from .core_logic import CoreLogic


class VectorizedCoreLogic(CoreLogic):
    def __init__(self, backend, n_envs):
        self._converter = DataConversion.get_converter(backend)
        self._n_envs = n_envs
        self._running_envs = self._converter.zeros(n_envs, dtype=bool)

        super().__init__()

    def get_mask(self, last):
        mask = self._converter.ones(self._n_envs, dtype=bool)
        terminated_episodes = (last & self._running_envs).sum()
        running_episodes = (~last & self._running_envs).sum()

        if running_episodes == 0 and terminated_episodes == 0:
            terminated_episodes = self._n_envs

        max_runs = terminated_episodes

        if self._n_episodes is not None:
            missing_episodes_move = self._n_episodes - self._total_episodes_counter - running_episodes

            max_runs = min(missing_episodes_move, max_runs)

        if self._n_episodes_per_fit is not None:
            missing_episodes_fit = self._n_episodes_per_fit - self._current_episodes_counter - running_episodes
            max_runs = min(missing_episodes_fit, max_runs)

        new_mask = self._converter.ones(terminated_episodes, dtype=bool)
        new_mask[max_runs:] = False
        mask[last] = new_mask

        self._running_envs = mask.copy()

        return mask

    def get_initial_state(self, initial_states):
        if initial_states is None or self._total_episodes_counter >= self._n_episodes:
            initial_state = None
        else:
            initial_state = initial_states[self._total_episodes_counter]  # FIXME

        return initial_state

    def after_step(self, last):
        n_active_envs = self._running_envs.sum()
        self._total_steps_counter += n_active_envs
        self._current_steps_counter += n_active_envs
        self._steps_progress_bar.update(n_active_envs)

        completed = last.sum()
        self._total_episodes_counter += completed
        self._current_episodes_counter += completed
        self._episodes_progress_bar.update(last.sum())

    def after_fit(self):
        super().after_fit()
        if self._n_episodes_per_fit is not None:
            self._running_envs = self._converter.zeros(self._n_envs, dtype=bool)

    def _reset_counters(self):
        super()._reset_counters()
        self._running_envs = self._converter.zeros(self._n_envs, dtype=bool)
