import torch
import numpy as np

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


def test_serialization(tmpdir):
    TorchUtils.set_default_device('cpu')

    a = DummyClass()
    a.save(tmpdir / 'test.msh')

    b = MushroomObject.load(tmpdir / 'test.msh')

    assert a == b
    assert b.not_saved == None


def test_serialization_lists(tmpdir):
    TorchUtils.set_default_device('cpu')

    a = DummyListClass()
    a.save(tmpdir / 'test_lists.msh')

    b = MushroomObject.load(tmpdir / 'test_lists.msh')

    assert a == b
    assert isinstance(b.numpy_list, list) and isinstance(b.numpy_list[0], np.ndarray)
    assert isinstance(b.numpy_nested[0], list) and isinstance(b.numpy_nested[0][0], np.ndarray)
    assert isinstance(b.torch_list, list) and isinstance(b.torch_list[0], torch.Tensor)
    assert isinstance(b.mushroom_list[0], DummyLeaf)
    assert isinstance(b.mushroom_nested[0], list) and isinstance(b.mushroom_nested[0][0], DummyLeaf)
    
    
def test_serialization_cuda_cpu(tmpdir):
    if torch.cuda.is_available():
        TorchUtils.set_default_device('cuda')

        a = DummyClass()
        a.save(tmpdir / 'test.msh')

        TorchUtils.set_default_device('cpu')

        assert a.torch_tensor.device.type == 'cuda'
        
        b = MushroomObject.load(tmpdir / 'test.msh')
        
        assert b.torch_tensor.device.type == 'cpu'

        assert a == b


def test_serialization_cpu_cuda(tmpdir):
    if torch.cuda.is_available():
        TorchUtils.set_default_device('cpu')

        a = DummyClass()
        a.save(tmpdir / 'test.msh')

        TorchUtils.set_default_device('cuda')

        assert a.torch_tensor.device.type == 'cpu'

        b = MushroomObject.load(tmpdir / 'test.msh')

        assert b.torch_tensor.device.type == 'cuda'

        assert a == b

        TorchUtils.set_default_device('cpu')


        
        
    


