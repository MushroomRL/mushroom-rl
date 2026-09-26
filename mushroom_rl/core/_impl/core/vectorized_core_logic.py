from mushroom_rl.core.array_backend import ArrayBackend

from .core_logic import CoreLogic


class VectorizedCoreLogic(CoreLogic):
    def __init__(self, backend, n_envs, device=None):
        self._array_backend = ArrayBackend.get_array_backend(backend)
        self._n_envs = n_envs
        self._device = device
        self._running_envs = self._array_backend.zeros(n_envs, dtype=bool, device=device)
        self._n_active_envs = 0
        self._n_reset_envs = 0
        self._n_completed_episodes = 0
        self._started_counter = 0

        super().__init__()

    def initialize_learn(self, n_steps_per_fit, n_episodes_per_fit):
        assert n_steps_per_fit is None or n_steps_per_fit >= self._n_envs, \
            "n_steps_per_fit cannot be lower than the number of environments."

        super().initialize_learn(n_steps_per_fit, n_episodes_per_fit)

    def get_mask(self, last):
        terminated_episodes = self._n_completed_episodes
        running_episodes = self._n_active_envs - terminated_episodes

        first_batch = running_episodes == 0 and terminated_episodes == 0

        if first_batch:
            mask = self._array_backend.ones(self._n_envs, dtype=bool, device=self._device)
            terminated_episodes = self._n_envs
        else:
            mask = self._running_envs

        max_runs = terminated_episodes
        n_continuing = running_episodes

        if self._n_episodes is not None:
            missing_episodes_move = max(self._n_episodes - self._total_episodes_counter - running_episodes, 0)
            max_runs = min(missing_episodes_move, max_runs)

        if self._n_episodes_per_fit is not None:
            missing_episodes_fit = max(self._n_episodes_per_fit - self._current_episodes_counter - running_episodes, 0)
            max_runs = min(missing_episodes_fit, max_runs)

        if self._n_steps is not None:
            missing_steps = max(self._n_steps - self._total_steps_counter, 0)
            n_continuing = min(missing_steps, n_continuing)
            max_runs = min(missing_steps - n_continuing, max_runs)

        new_mask = self._array_backend.ones(terminated_episodes, dtype=bool, device=self._device)
        new_mask[max_runs:] = False

        if first_batch:
            mask = new_mask
        else:
            mask[last & self._running_envs] = new_mask

            if n_continuing < running_episodes:
                continuing_mask = self._array_backend.ones(running_episodes, dtype=bool, device=self._device)
                continuing_mask[n_continuing:] = False
                mask[~last & self._running_envs] = continuing_mask

        self._running_envs = self._array_backend.copy(mask)
        self._n_active_envs = n_continuing + max_runs
        self._n_reset_envs = max_runs

        return mask

    def get_initial_state(self, initial_states, reset_mask):
        if initial_states is None:
            return None

        n_reset = self._n_reset_envs
        selected = initial_states[self._started_counter:self._started_counter + n_reset]
        self._started_counter += n_reset

        return self._array_backend.masked_init(reset_mask, selected, device=self._device)

    def after_step(self, last):
        self._total_steps_counter += self._n_active_envs
        self._current_steps_counter += self._n_active_envs
        self._steps_progress_bar.update(self._n_active_envs)

        completed = last.sum().item()
        self._n_completed_episodes = completed
        self._total_episodes_counter += completed
        self._current_episodes_counter += completed
        self._episodes_progress_bar.update(completed)

        return completed

    def after_fit_vectorized(self, last, n_carry_forward_steps):
        super().after_fit(n_carry_forward_steps)
        if self._n_episodes_per_fit is not None:
            self._running_envs = self._array_backend.zeros(self._n_envs, dtype=bool, device=self._device)
            self._n_active_envs = 0
            self._n_reset_envs = 0
            self._n_completed_episodes = 0
            return self._array_backend.ones(self._n_envs, dtype=bool, device=self._device)
        else:
            return last

    @property
    def converter(self):
        return self._array_backend

    @property
    def n_active_envs(self):
        return self._n_active_envs

    @property
    def n_reset_envs(self):
        return self._n_reset_envs

    def _reset_counters(self):
        super()._reset_counters()
        self._running_envs = self._array_backend.zeros(self._n_envs, dtype=bool, device=self._device)
        self._n_active_envs = 0
        self._n_reset_envs = 0
        self._n_completed_episodes = 0
        self._started_counter = 0
