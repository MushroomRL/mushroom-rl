import numpy as np
import pytest
import torch

from helper.arrays import assert_equal, make_array, to_numpy
from helper.cases import backend_devices, needs_cuda
from helper.utils import TestUtils as tu
from mushroom_rl.utils.priority_sampler import PrioritySampler, SumTree, PrefixSumSampler


def sampler_cases():
    return [pytest.param(impl, *case.values, id=f'{impl.__name__}-{case.id}', marks=case.marks)
            for impl in (SumTree, PrefixSumSampler) for case in backend_devices('numpy', 'torch')]


@pytest.mark.parametrize('impl, backend, device', sampler_cases())
def test_update(impl, backend, device, tmpdir):
    def array(data):
        return make_array(backend, data, device)

    n = 5
    priorities = np.array([1.0, 2.0, 1.0, 1.5, 0.5])

    sampler = impl(max_size=20, backend=backend, device=device)
    sampler.update(array(np.arange(n)), array(priorities))

    assert np.isclose(to_numpy(sampler.total_priority), 6.0)
    assert np.isclose(to_numpy(sampler.max_priority), 2.0)

    sampler.save(tmpdir / 'sampler.msh')
    loaded = PrioritySampler.load(tmpdir / 'sampler.msh', device=device)
    tu.assert_eq(sampler, loaded)
    assert np.isclose(to_numpy(loaded.total_priority), 6.0)

    n = 5

    sampler = impl(max_size=20, backend=backend, device=device)
    sampler.update(array(np.arange(n)), array(np.ones(n)))
    sampler.update(array([0]), array([5.0]))

    assert np.isclose(to_numpy(sampler.total_priority), 9.0)
    assert np.isclose(to_numpy(sampler.max_priority), 5.0)
    assert np.isclose(to_numpy(sampler.get(array([0.5]))[1]), 5.0)

    n = 5
    priorities = np.array([1.0, 2.0, 1.0, 1.5, 0.5])

    sampler = impl(max_size=20, backend=backend, device=device)
    sampler.update(array(np.arange(n)), array(priorities))

    idx, p = sampler.get(array([0.5, 3.0]))
    assert_equal(idx, [0, 2], backend, device)
    assert np.allclose(to_numpy(p), [1.0, 1.0])


@pytest.mark.parametrize('impl, backend, device', sampler_cases())
def test_mask(impl, backend, device):
    def array(data):
        return make_array(backend, data, device)

    n = 8

    sampler = impl(max_size=n, backend=backend, device=device)
    sampler.update(array(np.arange(n)), array(np.ones(n)))
    sampler.set_mask(array([3, 4]), array([True, True]))

    assert np.isclose(to_numpy(sampler.total_priority), 6.0)

    np.random.seed(0)
    torch.manual_seed(0)
    drawn, _ = sampler.sample(5000)

    assert set(to_numpy(drawn).tolist()).isdisjoint({3, 4})

    n = 8

    sampler = impl(max_size=n, backend=backend, device=device)
    sampler.update(array(np.arange(n)), array(np.ones(n)))
    sampler.set_mask(array([3]), array([True]))

    assert np.isclose(to_numpy(sampler.total_priority), 7.0)

    sampler.update(array([3]), array([2.0]))

    assert np.isclose(to_numpy(sampler.total_priority), 9.0)
    assert np.isclose(to_numpy(sampler.max_priority), 2.0)

    sampler.set_mask(array([5, 3, 5]), array([True, True, True]))

    assert np.isclose(to_numpy(sampler.total_priority), 6.0)
    assert np.isclose(to_numpy(sampler.max_priority), 1.0)

    sampler.set_mask(array([5, 3, 5]), array([True, False, True]))

    assert np.isclose(to_numpy(sampler.total_priority), 8.0)
    assert np.isclose(to_numpy(sampler.max_priority), 2.0)

    sampler.set_mask(array([0]), array([False]))

    assert np.isclose(to_numpy(sampler.total_priority), 8.0)

    sampler.set_mask(array([0]), array([True]))

    assert_equal(sampler.get(array([0.0, 2.0]))[0], [1, 3], backend, device)


@pytest.mark.parametrize('impl, backend, device', sampler_cases())
def test_sample(impl, backend, device):
    n = 12

    sampler = impl(max_size=n, backend=backend, device=device)
    sampler.update(make_array(backend, np.arange(n), device), make_array(backend, np.ones(n), device))

    np.random.seed(0)
    torch.manual_seed(0)
    drawn, p = sampler.sample(n)

    assert_equal(np.sort(to_numpy(drawn)), np.arange(n), None)
    assert_equal(p, np.ones(n), backend, device)
    assert p.dtype == (torch.float32 if backend == 'torch' else np.float64)


def test_dispatch(monkeypatch):
    assert type(PrioritySampler(100_000)) is PrefixSumSampler
    assert type(PrioritySampler(100_001)) is SumTree
    assert type(PrioritySampler(500_000, backend='torch', device='cpu')) is PrefixSumSampler
    assert type(PrioritySampler(500_001, backend='torch', device='cpu')) is SumTree

    monkeypatch.setitem(PrioritySampler.PREFIX_SUM_MAX_SIZE, 'numpy', 10)

    assert type(PrioritySampler(10)) is PrefixSumSampler
    assert type(PrioritySampler(11)) is SumTree

    calls = [
        lambda: PrioritySampler.get(None, None),
        lambda: PrioritySampler.sample(None, 1),
        lambda: PrioritySampler.update(None, None, None),
        lambda: PrioritySampler.set_mask(None, None, None),
        lambda: PrioritySampler.max_priority.fget(None),
        lambda: PrioritySampler.total_priority.fget(None),
    ]

    for call in calls:
        with pytest.raises(NotImplementedError):
            call()


@needs_cuda()
def test_dispatch_cuda():
    assert type(PrioritySampler(500_001, backend='torch', device='cuda')) is PrefixSumSampler
