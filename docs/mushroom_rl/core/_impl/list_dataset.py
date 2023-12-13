from copy import deepcopy

from mushroom_rl.core.serialization import Serializable


class ListDataset(Serializable):
    def __init__(self, is_stateful):
        self._dataset = list()
        self._policy_dataset = list()
        self._is_stateful = is_stateful

        self._add_save_attr(
            _dataset='pickle',
            _policy_dataset='pickle',
            _is_stateful='primitive'
        )

    @classmethod
    def from_array(cls, states, actions, rewards, next_states, absorbings, lasts, policy_states=None,
                   policy_next_states=None):
        is_stateful = (policy_states is not None) and (policy_next_states is not None)

        dataset = cls(is_stateful)

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

    def append(self, *step):
        step_copy = deepcopy(step)
        self._dataset.append(step_copy[:6])
        if self._is_stateful:
            self._policy_dataset.append(step_copy[6:])

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
        last_step = result._dataset[-1]
        modified_last_step = last_step[:-1] + (True,)
        result._dataset[-1] = modified_last_step
        result._dataset = result._dataset + other._dataset
        result._policy_dataset = result._policy_dataset + other._policy_dataset

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
    def n_episodes(self):
        n_episodes = 0
        for sample in self._dataset:
            if sample[5] is True:
                n_episodes += 1
        if self._dataset[-1][5] is not True:
            n_episodes += 1

        return n_episodes

