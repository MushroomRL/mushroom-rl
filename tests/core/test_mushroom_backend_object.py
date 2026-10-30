import numpy as np
import pytest
import torch

from mushroom_rl.core import MushroomObject, MDPInfo, AgentInfo, DatasetInfo
from mushroom_rl.core.array_backend import NumpyBackend, TorchBackend
from mushroom_rl.core.spaces import Box
from mushroom_rl.core._impl.extra_info import StepInfo
from mushroom_rl.rl_utils.running_stats import RunningStandardization, RunningAveragedWindow
from mushroom_rl.utils import TorchUtils


def make_dataset_info(env_backend, agent_backend, env_device=None, agent_device=None):
    return DatasetInfo(env_backend=env_backend, agent_backend=agent_backend, env_device=env_device,
                       agent_device=agent_device, horizon=10, gamma=0.9, state_shape=(1,), state_dtype=np.float64,
                       action_shape=(1,), action_dtype=np.float64, policy_state_shape=None)


def test_backend_object_numpy_has_no_device():
    stats = RunningStandardization((2,), backend='numpy')

    assert stats.backend is NumpyBackend
    assert stats.backend_name == 'numpy'
    assert stats.device is None


def test_backend_object_resolves_the_default_device_at_construction():
    TorchUtils.set_default_device('cpu')

    stats = RunningStandardization((2,), backend='torch')

    assert stats.backend is TorchBackend
    assert stats.backend_name == 'torch'
    assert stats.device == 'cpu'


def test_backend_object_rejects_a_device_without_device_support():
    with pytest.raises(ValueError):
        RunningStandardization((2,), backend='numpy', device='cpu')

    with pytest.raises(ValueError):
        AgentInfo(is_episodic=False, policy_state_shape=None, backend='numpy', device='cpu')

    box = Box(np.zeros(1), np.ones(1))
    with pytest.raises(ValueError):
        MDPInfo(box, box, 0.9, 10, backend='numpy', device='cpu')

    with pytest.raises(ValueError):
        make_dataset_info('numpy', 'numpy', env_device='cpu')


def test_backend_object_loads_on_the_default_device(tmpdir):
    TorchUtils.set_default_device('cpu')

    window = RunningAveragedWindow((2,), window_size=3, backend='torch')
    window.update_stats(torch.tensor([1., 2.]))
    window.save(tmpdir / 'window.msh')

    loaded = MushroomObject.load(tmpdir / 'window.msh')

    assert loaded.device == 'cpu'
    assert all(value.device.type == 'cpu' for value in loaded._avg_buffer)
    assert torch.equal(loaded.mean, torch.tensor([.5, 1.]))


def test_backend_object_loads_on_the_requested_device(tmpdir):
    if not torch.cuda.is_available():
        pytest.skip('requires cuda')

    TorchUtils.set_default_device('cpu')

    window = RunningAveragedWindow((2,), window_size=3, backend='torch')
    window.update_stats(torch.tensor([1., 2.]))
    window.save(tmpdir / 'window.msh')

    info = StepInfo(1, 'torch')
    info.append({'x': torch.tensor([1., 2.])})
    info.parse()
    info.save(tmpdir / 'info.msh')

    loaded_window = MushroomObject.load(tmpdir / 'window.msh', device='cuda')
    loaded_info = MushroomObject.load(tmpdir / 'info.msh', device='cuda')

    assert loaded_window.device == 'cuda'
    assert [value.device.type for value in loaded_window._avg_buffer] == ['cuda', 'cuda']
    assert torch.equal(torch.stack(list(loaded_window._avg_buffer)).cpu(), torch.tensor([[0., 0.], [1., 2.]]))

    assert loaded_info.device == 'cuda'
    assert loaded_info.parse()['x'].device.type == 'cuda'
    assert torch.equal(loaded_info.parse()['x'].cpu(), torch.tensor([[1., 2.]]))


def test_backend_object_secondary_devices_follow_the_load_device(tmpdir):
    TorchUtils.set_default_device('cpu')

    info = make_dataset_info('torch', 'numpy')
    assert info.env_device == 'cpu'
    assert info.agent_device is None

    info.save(tmpdir / 'info.msh')
    loaded = MushroomObject.load(tmpdir / 'info.msh')

    assert loaded.env_backend is TorchBackend
    assert loaded.env_backend_name == 'torch'
    assert loaded.env_device == 'cpu'
    assert loaded.agent_device is None

    if torch.cuda.is_available():
        loaded = MushroomObject.load(tmpdir / 'info.msh', device='cuda')

        assert loaded.env_device == 'cuda'
        assert loaded.agent_device is None


def test_step_info_to_backend_on_the_same_device_is_a_no_op():
    TorchUtils.set_default_device('cpu')

    info = StepInfo(1, 'torch')
    info.append({'x': torch.tensor([1., 2.])})

    same = info.to_backend('torch')
    other = info.to_backend('numpy')

    assert same._target_backend_name is None
    assert other._target_backend_name == 'numpy'
    assert other._target_device is None


def test_step_info_pending_target_device_follows_the_load_device(tmpdir):
    TorchUtils.set_default_device('cpu')

    info = StepInfo(1, 'numpy').to_backend('torch')
    info.append({'x': np.array([1., 2.])})
    info.save(tmpdir / 'info.msh')

    loaded = MushroomObject.load(tmpdir / 'info.msh')

    assert loaded._target_backend_name == 'torch'
    assert loaded._target_device == 'cpu'
    assert loaded.parse()['x'].device.type == 'cpu'

    if torch.cuda.is_available():
        loaded = MushroomObject.load(tmpdir / 'info.msh', device='cuda')

        assert loaded._target_device == 'cuda'
        assert loaded.parse()['x'].device.type == 'cuda'
