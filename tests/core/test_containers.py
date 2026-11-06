import numpy as np
import pytest
import torch

from helper.arrays import assert_equal, make_array
from helper.cases import backend_devices

from mushroom_rl.core._impl.containers import Container


def make_container(backend, device, capacity=8):
    dtypes = dict(numpy=[float, float, bool], torch=[torch.float, torch.float, torch.bool], list=[None] * 3)
    return Container.create(backend, [(capacity, 2), (capacity,), (capacity,)], dtypes[backend], device=device)


def fill(container, backend, device, rewards, lasts=None):
    lasts = [False] * len(rewards) if lasts is None else lasts
    for reward, last in zip(rewards, lasts):
        container.append(make_array(backend, [reward, reward], device), reward, last)
    return container


@pytest.mark.parametrize('backend, device', backend_devices())
def test_append(backend, device):
    container = make_container(backend, device)
    state = make_array(backend, [0.0, 1.0], device)
    container.append(state, 0.5, False)
    container.append(make_array(backend, [2.0, 3.0], device), 1.5, True)
    state[0] = 99.0

    assert len(container) == 2
    assert container.n_columns == 3
    assert len(container.data) == 3
    assert container.capacity == (None if backend == 'list' else 8)
    assert_equal(container.column(0), [[0.0, 1.0], [2.0, 3.0]], backend, device)
    assert_equal(container.column(1), [0.5, 1.5], backend, device)
    assert_equal(container.column(2), [False, True], backend, device)

    step = container[1]

    assert_equal(step[0], [2.0, 3.0])
    assert step[1] == 1.5
    assert bool(step[2])

    container.clear()

    assert len(container) == 0
    assert container.n_columns == 3


@pytest.mark.parametrize('backend, device', backend_devices())
def test_view(backend, device):
    container = fill(make_container(backend, device), backend, device, [0.0, 1.0, 2.0, 3.0])

    view = container.get_view(slice(1, 3))

    assert len(view) == 2
    assert_equal(view.column(1), [1.0, 2.0], backend, device)

    index = [0, 3] if backend == 'list' else make_array(backend, [0, 3], device)
    assert_equal(container.get_view(index).column(1), [0.0, 3.0], backend, device)

    view = container.get_view(slice(0, 2), copy=True)
    view.column(0)[0][0] = 99.0

    assert len(view) == 2
    assert_equal(container.column(0)[0], [0.0, 0.0])


@pytest.mark.parametrize('backend, device', backend_devices())
def test_concatenate(backend, device):
    a = fill(make_container(backend, device), backend, device, [0.0, 1.0], [False, True])
    b = fill(make_container(backend, device), backend, device, [2.0], [True])

    result = a + b

    assert len(result) == 3
    assert_equal(result.column(1), [0.0, 1.0, 2.0], backend, device)
    assert_equal(a.column(1), [0.0, 1.0], backend, device)

    a = fill(make_container(backend, device), backend, device, [0.0])
    b = fill(make_container(backend, device), backend, device, [1.0, 2.0], [True, True])
    a.append_batch(b)

    assert len(a) == 3
    assert_equal(a.column(1), [0.0, 1.0, 2.0], backend, device)

    full = fill(make_container(backend, device, capacity=2), backend, device, [0.0, 1.0])
    if backend == 'list':
        full.append_batch(b)
        assert len(full) == 4
    else:
        with pytest.raises(AssertionError):
            full.append_batch(b)


@pytest.mark.parametrize('backend, device', backend_devices())
def test_setitem(backend, device):
    container = fill(make_container(backend, device), backend, device, [0.0, 1.0, 2.0])
    row = (make_array(backend, [10.0, 11.0], device), 10.0, True)
    if backend == 'list':
        rows = ([np.array([20.0, 21.0]), np.array([22.0, 23.0])], [20.0, 22.0], [True, False])
    else:
        rows = (make_array(backend, [[20.0, 21.0], [22.0, 23.0]], device), make_array(backend, [20.0, 22.0], device),
                make_array(backend, [True, False], device))

    container[1] = row
    container[0:3:2] = rows
    row[0][0] = -1.0

    assert len(container) == 3
    assert_equal(container.column(0), [[20.0, 21.0], [10.0, 11.0], [22.0, 23.0]], backend, device)
    assert_equal(container.column(1), [20.0, 10.0, 22.0], backend, device)
    assert_equal(container.column(2), [True, True, False], backend, device)
    assert container[-1][1] == 22.0
    assert_equal(container[1:8][1], [10.0, 22.0])

    zeros = (make_array(backend, [[0.0, 0.0]] * 2, device), make_array(backend, [0.0] * 2, device),
             make_array(backend, [False] * 2, device))
    with pytest.raises(IndexError):
        container[3] = row
    with pytest.raises(IndexError):
        container[2:4] = zeros
    with pytest.raises(TypeError):
        container[make_array(backend, [0, 1], device)] = zeros

    with pytest.raises(IndexError):
        container[3]

    assert len(container) == 3


@pytest.mark.parametrize('backend, device', backend_devices())
def test_reserve(backend, device):
    container = make_container(backend, device, capacity=2)
    container.append(make_array(backend, [0.0, 1.0], device), 0.5, False)
    container.append(make_array(backend, [2.0, 3.0], device), 1.5, True)

    container.reserve(6)

    assert container.capacity == (None if backend == 'list' else 6)
    assert len(container) == 2
    assert_equal(container.column(0), [[0.0, 1.0], [2.0, 3.0]], backend, device)
    assert_equal(container.column(2), [False, True], backend, device)

    container.append(make_array(backend, [4.0, 5.0], device), 2.5, True)
    container.reserve(4)

    assert container.capacity == (None if backend == 'list' else 6)
    assert_equal(container.column(1), [0.5, 1.5, 2.5], backend, device)


@pytest.mark.parametrize('backend, device', backend_devices())
def test_n_episodes(backend, device):
    closed = fill(make_container(backend, device), backend, device, [0.0] * 4, [False, True, False, True])
    open_ = fill(make_container(backend, device), backend, device, [0.0] * 3)

    assert closed.n_episodes(2) == 2
    assert closed.n_episodes(2, skip_incomplete=False) == 2
    assert open_.n_episodes(2) == 0
    assert open_.n_episodes(2, skip_incomplete=False) == 1
    assert make_container(backend, device).n_episodes(2) == 0


@pytest.mark.parametrize('backend, device', backend_devices())
def test_from_array(backend, device):
    states = make_array(backend, [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], device)
    rewards = make_array(backend, [0.0, 1.0, 2.0], device)
    lasts = make_array(backend, [False, False, True], device)
    if backend == 'list':
        states, rewards, lasts = list(states), list(rewards), list(lasts)

    container = Container.from_array([states, rewards, lasts], device=device, backend=backend)

    assert len(container) == 3
    assert container.n_columns == 3
    assert_equal(container.column(0), states, backend, device)
    assert_equal(container.column(1), rewards, backend, device)
    assert container.n_episodes(2) == 1


@pytest.mark.parametrize('backend, device', backend_devices())
def test_n_envs(backend, device):
    container = Container.create(backend, [(4, 2)], [float], device=device, n_envs=2)
    container.append(make_array(backend, [10.0, 20.0, 30.0], device))
    container.append(make_array(backend, [1.0, 2.0, 3.0], device))

    assert container.n_envs == 2
    assert_equal(container.column(), [[10.0, 20.0], [1.0, 2.0]], backend, device)


def test_list_ragged():
    container = Container.create('list', [None] * 6, [None] * 6)
    container.append([0.0], 0, 0.0, [0.0, 1.0], False, False)
    container.append([0.0, 1.0, 2.0], 1, 1.0, [0.0], False, True)

    assert container.column(0) == [[0.0], [0.0, 1.0, 2.0]]
    assert container.n_episodes(5) == 1


def test_abstract():
    class Plain(Container):
        pass

    container = Plain('numpy')
    calls = [lambda: len(container), lambda: container[0], lambda: container.__setitem__(0, None),
             lambda: container + container, lambda: container.append(None), lambda: container.append_batch(container),
             container.clear, lambda: container.reserve(2), lambda: container.compact(0),
             lambda: container.get_view(0), container.column, lambda: container.n_episodes(0),
             lambda: container.data, lambda: container.n_columns, lambda: container.n_envs,
             lambda: container.capacity, lambda: Plain._allocate(None, None, None, None),
             lambda: Plain._from_array_impl(None, None)]

    assert Plain not in Container._by_backend.values()
    for call in calls:
        with pytest.raises(NotImplementedError):
            call()
