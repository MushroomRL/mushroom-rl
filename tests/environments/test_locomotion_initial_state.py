import numpy as np
import pytest


@pytest.mark.parametrize('name', ['Ant', 'HalfCheetah', 'Hopper', 'Walker2D'])
def test_scalar_locomotion_prescribed_state(name):
    envs = pytest.importorskip('mushroom_rl.environments.mujoco_envs')
    cls = getattr(envs, name)
    reference = cls(reset_noise_scale=0.0, n_substeps=1)
    target, _ = reference.reset()
    target.flags.writeable = False
    mdp = cls(n_substeps=1)
    try:
        actual, _ = mdp.reset(target)
        np.testing.assert_array_equal(actual, target)
        for _ in range(3):
            action = np.zeros(mdp.info.action_space.shape)
            actual, reward, absorbing, _ = mdp.step(action)
            expected, expected_reward, expected_absorbing, _ = reference.step(action)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
            assert reward == pytest.approx(expected_reward, abs=1e-12)
            assert absorbing == expected_absorbing
    finally:
        mdp.stop()
        reference.stop()


@pytest.mark.parametrize('name', ['AntWarp', 'HalfCheetahWarp', 'HopperWarp', 'Walker2DWarp'])
@pytest.mark.parametrize('partial', [False, True])
def test_warp_locomotion_prescribed_state(name, partial):
    envs = pytest.importorskip('mushroom_rl.environments.mujoco_warp_envs')
    import torch

    cls = getattr(envs, name)
    reference = cls(num_envs=2, device='cpu', reset_noise_scale=0.0, n_substeps=1)
    all_mask = torch.ones(2, dtype=torch.bool)
    target, _ = reference.reset_all(all_mask)
    target_copy = target.clone()
    mdp = cls(num_envs=2, device='cpu', n_substeps=1)
    try:
        before, _ = mdp.reset_all(all_mask)
        mask = torch.tensor([not partial, True])
        actual, _ = mdp.reset_all(mask, target)
        torch.testing.assert_close(actual[mask], target[mask], rtol=0, atol=0)
        torch.testing.assert_close(target, target_copy, rtol=0, atol=0)
        torch.testing.assert_close(actual[~mask], before[~mask], rtol=0, atol=0)
        if not partial:
            for _ in range(3):
                action = torch.zeros((2,) + mdp.info.action_space.shape)
                actual, reward, absorbing, _ = mdp.step_all(all_mask, action)
                expected, expected_reward, expected_absorbing, _ = reference.step_all(all_mask, action)
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
                torch.testing.assert_close(reward, expected_reward, rtol=0, atol=1e-6)
                assert torch.equal(absorbing, expected_absorbing)
    finally:
        mdp.stop()
        reference.stop()
