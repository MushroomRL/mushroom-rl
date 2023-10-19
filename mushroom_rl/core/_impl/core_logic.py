from tqdm import tqdm


class CoreLogic(object):
    def __init__(self):
        self.fit_required = None
        self.move_required = None

        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter = 0
        self._current_steps_counter = 0

        self._n_episodes = None
        self._n_steps_per_fit = None
        self._n_episodes_per_fit = None

        self._steps_progress_bar = None
        self._episodes_progress_bar = None

    def initialize_fit(self, n_steps_per_fit, n_episodes_per_fit):
        assert (n_episodes_per_fit is not None and n_steps_per_fit is None) \
               or (n_episodes_per_fit is None and n_steps_per_fit is not None)

        self._n_steps_per_fit = n_steps_per_fit
        self._n_episodes_per_fit = n_episodes_per_fit

        if n_steps_per_fit is not None:
            self.fit_required = lambda: self._current_steps_counter >= self._n_steps_per_fit
        else:
            self.fit_required = lambda: self._current_episodes_counter >= self._n_episodes_per_fit

    def initialize_evaluate(self):
        self.fit_required = lambda: False

    def initialize_run(self, n_steps, n_episodes, initial_states, quiet):
        assert n_episodes is not None and n_steps is None and initial_states is None\
            or n_episodes is None and n_steps is not None and initial_states is None\
            or n_episodes is None and n_steps is None and initial_states is not None

        self._n_episodes = len(initial_states) if initial_states is not None else n_episodes

        if n_steps is not None:
            self.move_required = lambda: self._total_steps_counter < n_steps

            self._steps_progress_bar = tqdm(total=n_steps,  dynamic_ncols=True, disable=quiet, leave=False)
            self._episodes_progress_bar = tqdm(disable=True)
        else:
            self.move_required = lambda: self._total_episodes_counter < self._n_episodes

            self._steps_progress_bar = tqdm(disable=True)
            self._episodes_progress_bar = tqdm(total=self._n_episodes, dynamic_ncols=True, disable=quiet, leave=False)

        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter = 0
        self._current_steps_counter = 0

    def get_initial_state(self, initial_states):
        if initial_states is None or self._total_episodes_counter == self._n_episodes:
            return None
        else:
            return initial_states[self._total_episodes_counter]

    def after_step(self, last):
        self._total_steps_counter += 1
        self._current_steps_counter += 1
        self._steps_progress_bar.update(1)

        if last:
            self._total_episodes_counter += 1
            self._current_episodes_counter += 1
            self._episodes_progress_bar.update(1)

    def after_fit(self):
        self._current_episodes_counter = 0
        self._current_steps_counter = 0

    def terminate_run(self):
        self._steps_progress_bar.close()
        self._episodes_progress_bar.close()