import numpy as np

from mushroom_rl.core import MushroomObject


class SumTree(MushroomObject):
    """
    This class implements a sum tree data structure.
    This is used, for instance, by ``PrioritizedReplayMemory``.

    The tree is kept in numpy on CPU regardless of the agent backend: it is a scalar,
    pointer-chasing structure whose per-node operations gain nothing from a tensor backend
    and would incur a host-device synchronization on each step if placed on the GPU.

    """
    def __init__(self, max_size):
        """
        Constructor.

        Args:
            max_size (int): maximum size of the tree.

        """
        self._max_size = max_size
        self._tree = np.zeros(2 * max_size - 1)
        self._masked = np.zeros(2 * max_size - 1, dtype=bool)

        super().__init__()
        self._add_save_attr(
            _max_size="primitive",
            _tree="numpy",
            _masked="numpy")

    def get(self, s):
        """
        Args:
            s (float): the value to query.

        Returns:
            The tree index and its priority.

        """
        idx = self._retrieve(s, 0)
        return idx, self._tree[idx]

    def update(self, idx, priorities):
        """
        Update the priorities at the given tree indices.

        Args:
            idx (np.ndarray): tree indices;
            priorities (np.ndarray): new priorities.

        """
        for i, p in zip(idx, priorities):
            delta = p - self._tree[i]
            self._tree[i] = p
            self._propagate(delta, i)
            if self._masked[i]:
                self._masked[i] = False
                self._propagate_mask(i)

    def mask(self, idx):
        """
        Exclude the leaves at the given tree indices from sampling, so :meth:`get` never returns them and they carry no
        weight in ``total_p``. A leaf is included again as soon as :meth:`update` writes a fresh priority to it.

        Args:
            idx (np.ndarray): tree indices of the leaves to exclude from sampling.

        """
        for i in idx:
            delta = -self._tree[i]
            self._tree[i] = 0.
            self._propagate(delta, i)
            if not self._masked[i]:
                self._masked[i] = True
                self._propagate_mask(i)

    def _propagate(self, delta, idx):
        parent_idx = (idx - 1) // 2
        self._tree[parent_idx] += delta

        if parent_idx != 0:
            self._propagate(delta, parent_idx)

    def _propagate_mask(self, idx):
        parent_idx = (idx - 1) // 2
        self._masked[parent_idx] = self._masked[2 * parent_idx + 1] and self._masked[2 * parent_idx + 2]

        if parent_idx != 0:
            self._propagate_mask(parent_idx)

    def _retrieve(self, s, idx):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self._tree):
            return idx

        if self._masked[left]:
            return self._retrieve(s, right)
        if self._masked[right]:
            return self._retrieve(s, left)

        if self._tree[left] == self._tree[right]:
            if s > self._tree[left]:
                s = s - self._tree[left]
            return self._retrieve(s, np.random.choice([left, right]))

        if s <= self._tree[left]:
            return self._retrieve(s, left)
        else:
            return self._retrieve(s - self._tree[left], right)

    @property
    def max_p(self):
        """
        Returns:
            The maximum priority among the ones in the tree.

        """
        return self._tree[-self._max_size:].max()

    @property
    def total_p(self):
        """
        Returns:
            The sum of the priorities in the tree, i.e. the value of the root node.

        """
        return self._tree[0]
