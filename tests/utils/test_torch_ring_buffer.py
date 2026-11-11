import torch

from mushroom_rl.utils.torch_utils import TorchRingBuffer


def test_torch_ring_buffer():
    buffer = TorchRingBuffer(3, 2, (2, ))

    for i in range(1, 5):
        buffer.push(torch.full((2, 2), float(i)))

    assert torch.equal(buffer.get(0), torch.full((2, 2), 4.))
    assert torch.equal(buffer.get(2), torch.full((2, 2), 2.))
    assert torch.equal(buffer.get(torch.tensor([0, 1])), torch.tensor([[4., 4.], [3., 3.]]))

    buffer.clear(torch.tensor([1]))

    assert torch.equal(buffer.get(torch.tensor([1, 2])), torch.tensor([[3., 3.], [0., 0.]]))
