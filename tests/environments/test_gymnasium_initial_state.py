import numpy as np
import pytest

gym = pytest.importorskip('gymnasium')

from mushroom_rl.environments.gymnasium_env import Gymnasium  # noqa: E402


@pytest.mark.parametrize('name', ['MountainCar-v0', 'MountainCarContinuous-v0'])
@pytest.mark.parametrize('initial', [(-0.75, -0.02), (-0.55, 0.01), (0.42, 0.04)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_supplied_gymnasium_state_controls_following_steps(name, initial, dtype):
    mdp = Gymnasium(name, headless=True)
    reference = gym.make(name, render_mode='rgb_array')
    state = np.array(initial, dtype=dtype)
    try:
        mdp.seed(17)
        observed, _ = mdp.reset(state)
        reference.reset(seed=17)
        reference.unwrapped.state = state.copy()
        np.testing.assert_array_equal(observed, state)
        assert not np.shares_memory(observed, state)

        for _ in range(12):
            action = np.array([2]) if name == 'MountainCar-v0' else np.array([0.5], dtype=np.float32)
            observed, reward, absorbing, _ = mdp.step(action)
            stock_action = action[0] if name == 'MountainCar-v0' else action
            expected, expected_reward, terminated, _, _ = reference.step(stock_action)
            np.testing.assert_array_equal(observed, expected)
            assert reward == expected_reward
            assert absorbing == terminated
            if terminated:
                break
        np.testing.assert_array_equal(state, np.array(initial, dtype=dtype))
    finally:
        mdp.stop()
        reference.close()
