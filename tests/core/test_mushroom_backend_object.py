from collections import deque, namedtuple

import numpy as np
import pytest
import torch

from helper.cases import devices

from mushroom_rl.core import Agent, MushroomObject, MDPInfo, AgentInfo, DatasetInfo
from mushroom_rl.core.array_backend import NumpyBackend, TorchBackend
from mushroom_rl.core.mushroom_backend_object import MushroomBackendObject
from mushroom_rl.core.spaces import Box
from mushroom_rl.core._impl.extra_info import StepInfo, EpisodeInfoParameters
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.preprocessors import StandardizationPreprocessor
from mushroom_rl.rl_utils.running_stats import RunningStandardization, RunningAveragedWindow


def make_dataset_info(env_backend, agent_backend, env_device=None, agent_device=None):
    return DatasetInfo(env_backend=env_backend, agent_backend=agent_backend, env_device=env_device,
                       agent_device=agent_device, horizon=10, gamma=0.9, state_shape=(1,), state_dtype=np.float64,
                       action_shape=(1,), action_dtype=np.float64, policy_state_shape=None)


def make_mdp_info(device=None):
    return MDPInfo(Box(np.zeros(2), np.ones(2)), Box(-np.ones(1), np.ones(1)), .99, 10, backend='torch',
                   device=device)


def load(path, device):
    return MushroomObject.load(path) if device == 'cpu' else MushroomObject.load(path, device=device)


def test_device_resolution():
    numpy_stats = RunningStandardization((2,), backend='numpy')
    torch_stats = RunningStandardization((2,), backend='torch')
    mdp_info = make_mdp_info()

    assert numpy_stats.backend is NumpyBackend
    assert numpy_stats.backend_name == 'numpy'
    assert numpy_stats.device is None
    assert torch_stats.backend is TorchBackend
    assert torch_stats.backend_name == 'torch'
    assert torch_stats.device == 'cpu'
    assert mdp_info.to('cpu') is mdp_info
    assert mdp_info.device == 'cpu'

    box = Box(np.zeros(1), np.ones(1))
    with pytest.raises(ValueError):
        RunningStandardization((2,), backend='numpy', device='cpu')
    with pytest.raises(ValueError):
        AgentInfo(is_episodic=False, policy_state_shape=None, backend='numpy', device='cpu')
    with pytest.raises(ValueError):
        MDPInfo(box, box, 0.9, 10, backend='numpy', device='cpu')
    with pytest.raises(ValueError):
        make_dataset_info('numpy', 'numpy', env_device='cpu')


@pytest.mark.parametrize('device', devices())
def test_load_device(tmpdir, device):
    window = RunningAveragedWindow((2,), window_size=3, backend='torch')
    window.update_stats(torch.tensor([1., 2.]))
    window.save(tmpdir / 'window.msh')

    info = StepInfo(1, 'torch')
    info.append({'x': torch.tensor([1., 2.])})
    info.parse()
    info.save(tmpdir / 'info.msh')

    loaded_window = load(tmpdir / 'window.msh', device)
    loaded_info = load(tmpdir / 'info.msh', device)

    assert loaded_window.device == device
    assert [value.device.type for value in loaded_window._avg_buffer] == [device, device]
    assert torch.equal(torch.stack(list(loaded_window._avg_buffer)).cpu(), torch.tensor([[0., 0.], [1., 2.]]))
    assert torch.equal(loaded_window.mean.cpu(), torch.tensor([.5, 1.]))
    assert loaded_info.device == device
    assert loaded_info.parse()['x'].device.type == device
    assert torch.equal(loaded_info.parse()['x'].cpu(), torch.tensor([[1., 2.]]))


@pytest.mark.parametrize('device', devices())
def test_load_secondary_devices(tmpdir, device):
    dataset_info = make_dataset_info('torch', 'numpy')

    assert dataset_info.env_device == 'cpu'
    assert dataset_info.agent_device is None

    dataset_info.save(tmpdir / 'dataset_info.msh')
    loaded = load(tmpdir / 'dataset_info.msh', device)

    assert loaded.env_backend is TorchBackend
    assert loaded.env_backend_name == 'torch'
    assert loaded.env_device == device
    assert loaded.agent_device is None

    info = StepInfo(1, 'numpy').to_backend('torch')
    info.append({'x': np.array([1., 2.])})
    info.save(tmpdir / 'info.msh')
    loaded = load(tmpdir / 'info.msh', device)

    assert loaded._target_backend_name == 'torch'
    assert loaded._target_device == device
    assert loaded.parse()['x'].device.type == device

    params = EpisodeInfoParameters(1, 'numpy').to_backend('torch')
    params.append(np.array([1., 2.]))
    params.save(tmpdir / 'params.msh')
    loaded = load(tmpdir / 'params.msh', device)

    assert loaded._target_device == device
    assert loaded.parse().device.type == device
    assert torch.equal(loaded.parse().cpu(), torch.tensor([[1., 2.]]))


@pytest.mark.parametrize('device', devices())
def test_agent_load_env_device(tmpdir, device):
    mdp_info = make_mdp_info(device)
    agent = Agent(mdp_info, EpsGreedy(.1, backend='torch'), backend='torch')
    agent.add_core_preprocessor(StandardizationPreprocessor(mdp_info))
    agent.save(tmpdir / 'agent.msh')

    loaded = Agent.load(tmpdir / 'agent.msh')
    loaded_on_env = Agent.load(tmpdir / 'agent.msh', env_device=device)

    assert loaded.mdp_info.device == 'cpu'
    assert loaded.core_preprocessors[0].device == 'cpu'
    assert loaded.core_preprocessors[0]._obs_runstand.mean.device.type == 'cpu'
    assert loaded_on_env.info.device == 'cpu'
    assert loaded_on_env.mdp_info.device == device
    assert loaded_on_env.core_preprocessors[0].device == device
    assert loaded_on_env.core_preprocessors[0]._obs_runstand.mean.device.type == device


@pytest.mark.parametrize('device', devices())
def test_move_to_device(device):
    pair = namedtuple('Pair', 'tensor number')
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.ones(2)
    value = dict(parameter=parameter, items=[torch.zeros(1), deque([torch.zeros(1)])], pair=pair(torch.zeros(1), 3),
                 plain=(torch.zeros(1), 'x'), other='x')

    class Holder(MushroomBackendObject):
        def __init__(self):
            super().__init__('torch')
            self._value = value
            self._full = torch.zeros(1)
            self._children = [RunningStandardization((2,), backend='torch'), None]
            self._add_save_attr(_value='torch', _full='torch!', _children='mushroom')

    holder = Holder().to(device)
    moved = holder._value

    assert moved['parameter'] is parameter
    assert parameter.device.type == device and parameter.grad.device.type == device
    assert moved['items'][0].device.type == device
    assert isinstance(moved['items'][1], deque) and moved['items'][1][0].device.type == device
    assert isinstance(moved['pair'], pair) and moved['pair'].tensor.device.type == device and moved['pair'].number == 3
    assert type(moved['plain']) is tuple and moved['plain'][0].device.type == device and moved['plain'][1] == 'x'
    assert moved['other'] == 'x'
    assert value['items'][0].device.type == 'cpu'
    assert holder.device == device
    assert holder._full.device.type == 'cpu'
    assert holder._children[0].device == device
    assert holder._children[1] is None
