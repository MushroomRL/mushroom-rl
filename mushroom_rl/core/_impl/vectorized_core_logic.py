from mushroom_rl.core import ArrayBackend
from .core_logic import CoreLogic


class VectorizedCoreLogic(CoreLogic):
    def __init__(self, backend, n_envs):
        self._array_backend = ArrayBackend.get_array_backend(backend)
        self._n_envs = n_envs
        self._running_envs = self._array_backend.zeros(n_envs, dtype=bool)

        super().__init__()

    def get_mask(self, last):
        terminated_episodes = (last & self._running_envs).sum().item()
        running_episodes = (~last & self._running_envs).sum().item()

        first_batch = running_episodes == 0 and terminated_episodes == 0

        if first_batch:
            mask = self._array_backend.ones(self._n_envs, dtype=bool)
            terminated_episodes = self._n_envs
        else:
            mask = self._running_envs

        max_runs = terminated_episodes

        if self._n_episodes is not None:
            missing_episodes_move = max(self._n_episodes - self._total_episodes_counter - running_episodes, 0)
            max_runs = min(missing_episodes_move, max_runs)

        if self._n_episodes_per_fit is not None:
            missing_episodes_fit = max(self._n_episodes_per_fit - self._current_episodes_counter - running_episodes, 0)
            max_runs = min(missing_episodes_fit, max_runs)

        new_mask = self._array_backend.ones(terminated_episodes, dtype=bool)
        new_mask[max_runs:] = False

        if first_batch:
            mask = new_mask
        else:
            mask[last] = new_mask

        self._running_envs = self._array_backend.copy(mask)

        return mask

    def get_initial_state(self, initial_states):
        if initial_states is None or self._total_episodes_counter >= self._n_episodes:
            initial_state = None
        else:
            initial_state = initial_states[self._total_episodes_counter]  # FIXME

        return initial_state

    def after_step(self, last):
        n_active_envs = self._running_envs.sum().item()
        self._total_steps_counter += n_active_envs
        self._current_steps_counter += n_active_envs
        self._steps_progress_bar.update(n_active_envs)

        completed = last.sum().item()
        self._total_episodes_counter += completed
        self._current_episodes_counter += completed
        self._episodes_progress_bar.update(completed)

    def after_fit_vectorized(self, last):
        super().after_fit()
        if self._n_episodes_per_fit is not None:
            self._running_envs = self._array_backend.zeros(self._n_envs, dtype=bool)
            return self._array_backend.ones(self._n_envs, dtype=bool)
        else:
            return last

    def _reset_counters(self):
        super()._reset_counters()
        self._running_envs = self._array_backend.zeros(self._n_envs, dtype=bool)

    @property
    def converter(self):
        return self._array_backend
