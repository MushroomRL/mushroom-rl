import numpy as np
import pytest
import torch

pytest.importorskip("mujoco_warp")
wp = pytest.importorskip("warp")

from mushroom_rl.environments.mujoco_warp_envs import AntWarp, HalfCheetahWarp, HopperWarp, Walker2DWarp  # noqa: E402


@pytest.mark.parametrize('env, name', [(AntWarp, 'ant'), (HalfCheetahWarp, 'half_cheetah'), (HopperWarp, 'hopper'),
                                       (Walker2DWarp, 'walker2d')])
def test_warp_locomotion(env, name):
    wp.set_device('cpu')
    torch.manual_seed(1)
    mdp = env(num_envs=4, use_graph_capture=False)
    mask = torch.ones(4, dtype=torch.bool, device='cpu')

    obs, _ = mdp.reset_all(mask)
    for _ in range(10):
        obs, reward, absorbing, _ = mdp.step_all(mask, torch.rand(4, mdp.info.action_space.shape[0], device='cpu'))

    obs_test = np.load(f'tests/environments/mujoco_warp_envs/data/test_warp_locomotion_{name}.npy')

    assert np.allclose(obs.numpy(), obs_test)
