import numpy as np
import torch

from mushroom_rl.core import ArrayBackend, MushroomBackendObject


class PrioritySampler(MushroomBackendObject):
    """
    Base class of the samplers storing a priority for each element and drawing elements with a probability
    proportional to it. Instantiating ``PrioritySampler`` builds the implementation chosen from the backend, the
    device and ``max_size``; the size thresholds are in ``PrioritySampler.PREFIX_SUM_MAX_SIZE``.

    """
    PREFIX_SUM_MAX_SIZE = dict(numpy=100_000, torch=500_000)

    def __new__(cls, max_size=None, *, backend='numpy', device=None):
        if cls is not PrioritySampler:
            return super().__new__(cls)

        array_backend = ArrayBackend.get_array_backend(backend)
        on_cuda = array_backend.supports_devices() and torch.device(array_backend.check_device(device)).type == 'cuda'

        if on_cuda or max_size <= PrioritySampler.PREFIX_SUM_MAX_SIZE[backend]:
            return super().__new__(PrefixSumSampler)
        else:
            return super().__new__(SumTree)

    def __init__(self, max_size=None, *, backend='numpy', device=None):
        """
        Constructor.

        Args:
            max_size (int, None): number of elements of the sampler;
            backend (str, 'numpy'): the backend of the indices and priorities exchanged with the sampler;
            device (str, None): device of the indices and priorities exchanged with the sampler, or ``None`` for the
                default one. Only allowed with the torch backend.

        """
        super().__init__(backend, device)

        self._max_size = max_size

        self._add_save_attr(_max_size='primitive')

    def get(self, values):
        """
        Retrieve, for each value, the element whose cumulative priority interval contains it.

        Args:
            values (Array): the values to retrieve, in ``[0, total_priority)``.

        Returns:
            The indices of the retrieved elements and their priorities.

        """
        raise NotImplementedError

    def sample(self, n_samples):
        """
        Draw elements with a probability proportional to their priority. Masked elements are never drawn.

        Args:
            n_samples (int): the number of elements to draw.

        Returns:
            The indices of the drawn elements and their priorities.

        """
        raise NotImplementedError

    def update(self, idx, priorities):
        """
        Set the priorities of the given elements, including them in sampling again.

        Args:
            idx (Array): the indices of the elements to update;
            priorities (Array): the new priorities.

        """
        raise NotImplementedError

    def set_mask(self, idx, masked):
        """
        Set whether the given elements are excluded from sampling. An excluded element keeps its priority, but it
        is never drawn and does not count in ``total_priority`` and ``max_priority``.

        Args:
            idx (Array): the indices of the elements to set;
            masked (Array): for each element, whether it is excluded from sampling.

        """
        raise NotImplementedError

    @property
    def max_priority(self):
        """
        Returns:
            The maximum priority among the elements that are not masked.

        """
        raise NotImplementedError

    @property
    def total_priority(self):
        """
        Returns:
            The sum of the priorities of the elements that are not masked.

        """
        raise NotImplementedError


class SumTree(PrioritySampler):
    """
    Priority sampler storing the priorities in a sum tree, with sampling and update logarithmic in ``max_size`` per
    element.

    """
    def __init__(self, max_size=None, **kwargs):
        super().__init__(max_size, **kwargs)

        self._nodes = np.zeros(2 * max_size - 1)
        self._masked = np.zeros(2 * max_size - 1, dtype=bool)

        self._add_save_attr(
            _nodes='numpy',
            _masked='numpy'
        )

    def get(self, values):
        leaves = np.array([self._retrieve(value, 0) for value in ArrayBackend.convert(values, to='numpy')], dtype=int)

        return self._to_backend(leaves - self._max_size + 1), self._to_backend(self._nodes[leaves])

    def sample(self, n_samples):
        segment_length = self.total_priority / n_samples

        return self.get(np.random.uniform(np.arange(n_samples) * segment_length,
                                          np.arange(1, n_samples + 1) * segment_length))

    def update(self, idx, priorities):
        leaves = ArrayBackend.convert(idx, to='numpy') + self._max_size - 1
        for i, p in zip(leaves, ArrayBackend.convert(priorities, to='numpy')):
            contribution = 0. if self._masked[i] else self._nodes[i]
            self._nodes[i] = p
            self._propagate(p - contribution, i)
            if self._masked[i]:
                self._masked[i] = False
                self._propagate_mask(i)

    def set_mask(self, idx, masked):
        leaves = ArrayBackend.convert(idx, to='numpy') + self._max_size - 1
        masked = ArrayBackend.convert(masked, to='numpy')

        for i in leaves[masked]:
            if not self._masked[i]:
                self._propagate(-self._nodes[i], i)
                self._masked[i] = True
                self._propagate_mask(i)

        for i in leaves[~masked]:
            if self._masked[i]:
                self._propagate(self._nodes[i], i)
                self._masked[i] = False
                self._propagate_mask(i)

    @property
    def max_priority(self):
        leaf_priorities = self._nodes[-self._max_size:]
        return float(np.where(self._masked[-self._max_size:], 0., leaf_priorities).max())

    @property
    def total_priority(self):
        return float(self._nodes[0])

    def _propagate(self, delta, node):
        parent = (node - 1) // 2
        self._nodes[parent] += delta

        if parent != 0:
            self._propagate(delta, parent)

    def _propagate_mask(self, node):
        parent = (node - 1) // 2
        self._masked[parent] = self._masked[2 * parent + 1] and self._masked[2 * parent + 2]

        if parent != 0:
            self._propagate_mask(parent)

    def _retrieve(self, value, node):
        left = 2 * node + 1
        right = left + 1

        if left >= len(self._nodes):
            return node

        if self._masked[left]:
            return self._retrieve(value, right)
        if self._masked[right]:
            return self._retrieve(value, left)

        if value < self._nodes[left]:
            return self._retrieve(value, left)
        else:
            return self._retrieve(value - self._nodes[left], right)

    def _to_backend(self, array):
        return ArrayBackend.convert(array, to=self.backend_name, device=self._device)


class PrefixSumSampler(PrioritySampler):
    """
    Priority sampler drawing elements from the prefix sums of the priorities, with sampling linear in ``max_size`` and
    update constant per element.

    """
    def __init__(self, max_size=None, **kwargs):
        super().__init__(max_size, **kwargs)

        dtype = self._backend.to_backend_dtype(torch.float64)
        self._priorities = self._backend.zeros(max_size, dtype=dtype, device=self._device)
        self._masked_priorities = self._backend.zeros(max_size, dtype=dtype, device=self._device)
        self._cumsum = None

        serialization = self._backend.get_backend_serialization()
        self._add_save_attr(
            _priorities=serialization,
            _masked_priorities=serialization,
            _cumsum='none'
        )

    def get(self, values):
        return self._retrieve(self._cumulative_priorities, self._to_backend(values))

    def sample(self, n_samples):
        cumsum = self._cumulative_priorities
        segment_length = cumsum[-1] / n_samples
        segment_idx = self._backend.arange(0, n_samples, dtype=cumsum.dtype, device=self._device)

        return self._retrieve(cumsum, self._backend.uniform(segment_idx * segment_length,
                                                            (segment_idx + 1) * segment_length, device=self._device))

    def update(self, idx, priorities):
        idx = self._to_backend(idx)
        priorities = self._to_backend(priorities)
        if self.backend_name == 'torch':
            priorities = priorities.double()
        self._priorities[idx] = priorities
        self._masked_priorities[idx] = priorities
        self._cumsum = None

    def set_mask(self, idx, masked):
        idx = self._to_backend(idx)
        self._masked_priorities[idx] = self._backend.where(self._to_backend(masked), 0., self._priorities[idx])
        self._cumsum = None

    @property
    def max_priority(self):
        return self._backend.max(self._masked_priorities)

    @property
    def total_priority(self):
        return self._cumulative_priorities[-1]

    def _retrieve(self, cumsum, values):
        idx = self._backend.clip(self._backend.searchsorted(cumsum, values, side='right'), 0, self._max_size - 1)

        priorities = self._priorities[idx]
        if self.backend_name == 'torch':
            priorities = priorities.float()

        return idx, priorities

    def _to_backend(self, array):
        return ArrayBackend.convert(array, to=self.backend_name, device=self._device)

    @property
    def _cumulative_priorities(self):
        if self._cumsum is None:
            self._cumsum = self._backend.cumsum(self._masked_priorities)

        return self._cumsum
