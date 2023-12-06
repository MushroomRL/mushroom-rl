import torch
import numpy as np

from mushroom_rl.core import Serializable
from mushroom_rl.utils import TorchUtils


class DummyClass(Serializable):
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


def test_serialization(tmpdir):
    TorchUtils.set_default_device('cpu')

    a = DummyClass()
    a.save(tmpdir / 'test.msh')
    
    b = Serializable.load(tmpdir / 'test.msh')
    
    assert a == b
    assert b.not_saved == None
    
    
def test_serialization_cuda_cpu(tmpdir):
    if torch.cuda.is_available():
        TorchUtils.set_default_device('cuda')

        a = DummyClass()
        a.save(tmpdir / 'test.msh')

        TorchUtils.set_default_device('cpu')

        assert a.torch_tensor.device.type == 'cuda'
        
        b = Serializable.load(tmpdir / 'test.msh')
        
        assert b.torch_tensor.device.type == 'cpu'

        assert a == b


def test_serialization_cpu_cuda(tmpdir):
    if torch.cuda.is_available():
        TorchUtils.set_default_device('cpu')

        a = DummyClass()
        a.save(tmpdir / 'test.msh')

        TorchUtils.set_default_device('cuda')

        assert a.torch_tensor.device.type == 'cpu'

        b = Serializable.load(tmpdir / 'test.msh')

        assert b.torch_tensor.device.type == 'cuda'

        assert a == b

        TorchUtils.set_default_device('cpu')


        
        
    


