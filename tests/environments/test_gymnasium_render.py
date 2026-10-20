import numpy as np
import pytest

gym = pytest.importorskip('gymnasium')

from mushroom_rl.environments.gymnasium_env import Gymnasium  # noqa: E402


@pytest.mark.parametrize('name', ['MountainCarContinuous-v0', 'Pendulum-v1'])
def test_gymnasium_frames_follow_steps_and_resets(name, monkeypatch):
    monkeypatch.setenv('SDL_VIDEODRIVER', 'dummy')
    monkeypatch.setenv('SDL_AUDIODRIVER', 'dummy')
    mdp = Gymnasium(name, headless=True)
    reference = gym.make(name, render_mode='rgb_array')
    try:
        for seed in [17, 43]:
            mdp.seed(seed)
            mdp.reset()
            reference.reset(seed=seed)
            for index, record in enumerate([True, False, True, True]):
                action = np.array([np.sin(index * 0.3)], dtype=np.float32)
                mdp.step(action)
                reference.step(action)
                frame = mdp.render(record=record)
                if record:
                    np.testing.assert_array_equal(frame, reference.render())
                else:
                    assert frame is None
            mdp.stop()
            reference.close()
    finally:
        mdp.stop()
        reference.close()
