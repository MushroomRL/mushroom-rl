from copy import deepcopy

from mushroom_rl.core.serialization import Serializable


class ListDataset(Serializable):
    def __init__(self, is_stateful, is_vectorized):
        self._dataset = list()
        self._policy_dataset = list()
        self._is_stateful = is_stateful

        if is_vectorized:
            self._mask = list()
        else:
            self._mask = None

        self._add_all_save_attr()

    @classmethod
    def create_new_instance(cls, dataset):
        """
        Creates an empty instance of the Dataset and populates essential data structures

        Args:
            dataset (ListDataset): a template dataset to be used to create the new instance.

        Returns:
            A new empty instance of the dataset.

        """

        new_dataset = cls.__new__(cls)

        new_dataset._dataset = None
        new_dataset._policy_dataset = None
        new_dataset._is_stateful = dataset._is_stateful
        new_dataset._mask = None

        new_dataset._add_all_save_attr()

        return new_dataset

    @classmethod
    def from_array(cls, states, actions, rewards, next_states, absorbings, lasts, policy_states=None,
                   policy_next_states=None):
        is_stateful = (policy_states is not None) and (policy_next_states is not None)

        dataset = cls(is_stateful, False)

        if dataset._is_stateful:
            for s, a, r, ss, ab, last, ps, pss in zip(states, actions, rewards, next_states,
                                                      absorbings.astype(bool), lasts.astype(bool),
                                                      policy_states, policy_next_states):
                dataset.append(s, a, r.item(), ss, ab.item(), last.item(), ps.item(), pss.item())
        else:
            for s, a, r, ss, ab, last in zip(states, actions, rewards, next_states,
                                             absorbings.astype(bool), lasts.astype(bool)):
                dataset.append(s, a, r.item(), ss, ab.item(), last.item())

        return dataset

    def __len__(self):
        return len(self._dataset)

    def append(self, *step, mask=None):
        step_copy = deepcopy(step)
        self._dataset.append(step_copy[:6])
        if self._is_stateful:
            self._policy_dataset.append(step_copy[6:])

        if mask is not None:
            self._mask.append(mask)

    def clear(self):
        self._dataset = list()

    def get_view(self, index, copy=False):
        view = self.create_new_instance(self)

        if isinstance(index, (int, slice)):
            view._dataset = self._dataset[index]
        else:
            view._dataset = [self._dataset[i] for i in index]

        if self._mask is not None:
            if isinstance(index, (int, slice)):
                view._mask = self._mask[index, ...]
            else:
                view._mask = [self._mask[i] for i in index]

        if copy:
            return view.copy()
        else:
            return view

    def __getitem__(self, index):
        return self._dataset[index]

    def __add__(self, other):
        result = self.create_new_instance(self)
        last_step = self._dataset[-1]
        modified_last_step = last_step[:-1] + (True,)
        result._dataset[-1] = modified_last_step
        result._dataset = self._dataset + other._dataset
        result._policy_dataset = self._policy_dataset + other._policy_dataset

        return result

    @property
    def state(self):
        return [step[0] for step in self._dataset]

    @property
    def action(self):
        return [step[1] for step in self._dataset]

    @property
    def reward(self):
        return [step[2] for step in self._dataset]

    @property
    def next_state(self):
        return [step[3] for step in self._dataset]

    @property
    def absorbing(self):
        return [step[4] for step in self._dataset]

    @property
    def last(self):
        return [step[5] for step in self._dataset]

    @property
    def policy_state(self):
        return [step[6] for step in self._dataset]

    @property
    def policy_next_state(self):
        return [step[7] for step in self._dataset]

    @property
    def is_stateful(self):
        return self._is_stateful

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, new_mask):
        self._mask = new_mask

    @property
    def n_episodes(self):
        n_episodes = 0
        for sample in self._dataset:
            if sample[5] is True:
                n_episodes += 1
        if self._dataset[-1][5] is not True:
            n_episodes += 1

        return n_episodes

    def _add_all_save_attr(self):
        self._add_save_attr(
            _dataset='pickle',
            _policy_dataset='pickle',
            _is_stateful='primitive'
        )

