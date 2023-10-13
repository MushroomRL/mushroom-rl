from copy import deepcopy

import numpy as np

from mushroom_rl.core.serialization import Serializable


class ListDataset(Serializable):
    def __init__(self):
        self._dataset = list()

        self._add_save_attr(
            _dataset='pickle'
        )

    @classmethod
    def from_array(cls, states, actions, rewards, next_states, absorbings, lasts):
        dataset = cls()

        for s, a, r, ss, ab, last in zip(states, actions, rewards, next_states,
                                         absorbings.astype(bool), lasts.astype(bool)
                                         ):
            dataset.append(s, a, r.item(), ss, ab.item(), last.item())

        return dataset

    def __len__(self):
        return len(self._dataset)

    def append(self, *step):
        assert len(step) == 6
        step_copy = deepcopy(step)
        self._dataset.append(step_copy)

    def clear(self):
        self._dataset = list()

    def get_view(self, index):
        view = self.copy()

        if isinstance(index, (int, slice)):
            view._dataset = self._dataset[index]
        else:
            view._dataset = [self._dataset[i] for i in index]

        return view

    def __getitem__(self, index):
        return self._dataset[index]

    def __add__(self, other):
        result = self.copy()
        result._dataset = self._dataset + other._dataset

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
