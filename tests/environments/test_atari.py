import numpy as np
import pytest

from helper.envs import discrete_action, rollout

pytest.importorskip('ale_py')

from mushroom_rl.environments.atari import Atari  # noqa: E402


def test_atari():
    np.random.seed(1)
    mdp = Atari(name='ALE/Pong-v5')
    mdp.seed(1)
    ns = rollout(mdp, discrete_action)

    assert np.allclose(ns, np.load('tests/environments/data/test_atari.npy'))


def test_render():
    mdp = Atari(name='ALE/Pong-v5', headless=True)
    mdp.reset()

    assert mdp.full_name() == 'Atari.ALE/Pong-v5'
    assert mdp.render(record=True).shape == (210, 160, 3)
    assert mdp.render() is None
