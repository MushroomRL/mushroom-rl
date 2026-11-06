import numpy as np
import pytest
import torch

from helper.cases import needs_cuda

from mushroom_rl.core import MushroomObject
from mushroom_rl.utils import TorchUtils


class DummyClass(MushroomObject):
    def __init__(self):
        self.torch_tensor = torch.randn(2, 2).to(TorchUtils.get_device())
        self.numpy_array = np.random.randn(3, 4)
        self.scalar = 1
        self.dictionary = {'a': 'test', 'b': 5, 'd': (2, 3)}
        self.not_saved = 'test2'

        self._add_save_attr(
            torch_tensor='torch',
            numpy_array='numpy',
            scalar='primitive',
            dictionary='pickle',
            not_saved='none'
        )

    def __eq__(self, other):
        f1 = torch.equal(self.torch_tensor.cpu(), other.torch_tensor.cpu())
        f2 = np.array_equal(self.numpy_array, other.numpy_array)
        f3 = self.scalar == other.scalar
        f4 = self.dictionary == other.dictionary

        return f1 and f2 and f3 and f4


class DummyLeaf(MushroomObject):
    def __init__(self, value=0):
        self.value = value
        self._add_save_attr(value='primitive')

    def __eq__(self, other):
        return isinstance(other, DummyLeaf) and self.value == other.value


class DummyMethod(MushroomObject):
    def __init__(self, method):
        self.data = {'a': [1, 2], 'b': 'text'}
        self._add_save_attr(data=method)


class DummyListClass(MushroomObject):
    def __init__(self):
        self.numpy_list = [np.arange(2.0), np.arange(3.0)]
        self.numpy_nested = [[np.zeros(2), np.ones(1)], [np.arange(3.0)]]
        self.torch_list = [torch.arange(2.0), torch.arange(3.0)]
        self.pickle_list = [[1, 2], [3, [4, 5]]]
        self.mushroom_list = [DummyLeaf(1), DummyLeaf(2)]
        self.mushroom_nested = [[DummyLeaf(1)], [DummyLeaf(2), DummyLeaf(3)]]

        self._add_save_attr(
            numpy_list='numpy',
            numpy_nested='numpy',
            torch_list='torch',
            pickle_list='pickle',
            mushroom_list='mushroom',
            mushroom_nested='mushroom'
        )

    @classmethod
    def _eq(cls, this, that):
        if isinstance(this, list):
            return isinstance(that, list) and len(this) == len(that) \
                and all(cls._eq(x, y) for x, y in zip(this, that))
        if isinstance(this, np.ndarray):
            return np.array_equal(this, that)
        if isinstance(this, torch.Tensor):
            return torch.equal(this.cpu(), that.cpu())
        return this == that

    def __eq__(self, other):
        attributes = ['numpy_list', 'numpy_nested', 'torch_list', 'pickle_list',
                      'mushroom_list', 'mushroom_nested']
        return all(self._eq(getattr(self, a), getattr(other, a)) for a in attributes)


@pytest.mark.parametrize('save_device, load_device', [('cpu', 'cpu'),
                                                      pytest.param('cuda', 'cpu', marks=needs_cuda()),
                                                      pytest.param('cpu', 'cuda', marks=needs_cuda())])
def test_serialization(tmpdir, save_device, load_device):
    TorchUtils.set_default_device(save_device)
    a = DummyClass()
    a.save(tmpdir / 'test.msh')

    TorchUtils.set_default_device(load_device)
    b = MushroomObject.load(tmpdir / 'test.msh')

    assert a.torch_tensor.device.type == save_device
    assert b.torch_tensor.device.type == load_device
    assert a == b
    assert b.not_saved is None


def test_serialization_lists(tmpdir):
    a = DummyListClass()
    a.save(tmpdir / 'test_lists.msh')

    b = MushroomObject.load(tmpdir / 'test_lists.msh')

    assert a == b
    assert isinstance(b.numpy_list, list) and isinstance(b.numpy_list[0], np.ndarray)
    assert isinstance(b.numpy_nested[0], list) and isinstance(b.numpy_nested[0][0], np.ndarray)
    assert isinstance(b.torch_list, list) and isinstance(b.torch_list[0], torch.Tensor)
    assert isinstance(b.mushroom_list[0], DummyLeaf)
    assert isinstance(b.mushroom_nested[0], list) and isinstance(b.mushroom_nested[0][0], DummyLeaf)


def test_save_methods(tmpdir):
    DummyMethod('json').save(tmpdir / 'json.msh')

    assert MushroomObject.load(tmpdir / 'json.msh').data == {'a': [1, 2], 'b': 'text'}

    with pytest.raises(NotImplementedError):
        DummyMethod('unknown').save(tmpdir / 'unknown.msh')
    with pytest.raises(ValueError):
        MushroomObject.load(tmpdir / 'missing.msh')


def test_logger_forwarding():
    class Node(MushroomObject):
        def __init__(self, child=None, other=None):
            self.child = child
            self.other = other
            self._add_logger_attr('child', 'other', group='inner')

    leaf = Node()
    root = Node(Node(leaf))
    root.set_logger('logger', prefix='outer')

    assert root.child._log_prefix == 'outer/inner'
    assert leaf._log_prefix == 'outer/inner/inner'
    assert leaf.logger == 'logger'
    assert root.other is None
