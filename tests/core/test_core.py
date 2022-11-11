import numpy as np

from mushroom_rl.core import Agent, Core
from mushroom_rl.environments import Atari

from mushroom_rl.policy import Policy


class RandomDiscretePolicy(Policy):
    def __init__(self, n):
        self._n = n

    def draw_action(self, state):
        return [np.random.randint(self._n)]


class DummyAgent(Agent):
    def __init__(self, mdp_info):
        policy = RandomDiscretePolicy(mdp_info.action_space.n)
        super().__init__(mdp_info, policy)

    def fit(self, dataset, **info):
        pass


def test_core():
    mdp = Atari(name='BreakoutDeterministic-v4')

    agent = DummyAgent(mdp.info)

    core = Core(agent, mdp)

    np.random.seed(2)
    mdp.seed(2)

    core.learn(n_steps=100, n_steps_per_fit=1)

    dataset, info = core.evaluate(n_steps=20, get_env_info=True)

    assert 'lives' in info
    assert 'episode_frame_number' in info
    assert 'frame_number' in info

    info_lives = np.array(info['lives'])

    print(info_lives)
    lives_gt = np.array([5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
    assert len(info_lives) == 20
    assert np.all(info_lives == lives_gt)
    assert len(dataset) == 20



