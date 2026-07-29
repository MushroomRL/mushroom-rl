"""
This script shows how to record a video of a MultiprocessEnvironment.

Every worker draws its own frames, so what a vectorized environment renders is a stack of images rather than
a single one. The recorder gives each environment a separate lossless temporary file and concatenates them,
one environment after the other, when the recording is stopped: asking for five episodes produces a single
video file showing the five episodes in sequence. Since the temporary files are lossless, the resulting video
is encoded only once and has the same quality of a video recorded from a single environment.

Nothing has to be configured for this to happen. The environments are rendered as usual, with the ``render``
and ``record`` flags of the evaluation, and the frame rate is taken from the environment by ``set_logger``.
Each worker draws in a window of its own, so the run opens one viewer per parallel copy. The policy is a
random one, as the recording does not depend on what the agent does.

"""
import numpy as np

from mushroom_rl.core import Agent, Core, Logger, MultiprocessEnvironment
from mushroom_rl.environments import CarOnHill
from mushroom_rl.policy import Policy
from mushroom_rl.utils import get_log_dir


class RandomPolicy(Policy):
    """
    Policy drawing an action uniformly at random, for every environment being run in parallel.

    """
    def __init__(self, action_space):
        self._n_actions = action_space.n

        super().__init__()

    def draw_action(self, state):
        return np.random.randint(self._n_actions, size=(len(state), 1))


def experiment(n_episodes_recorded, n_envs, seed=None):
    np.random.seed(seed)

    # MDP
    mdp = MultiprocessEnvironment(CarOnHill, n_envs=n_envs)
    mdp.seed(seed)

    logger = Logger('multiprocess_env_recording', results_dir=get_log_dir(__file__), use_timestamp=True)
    logger.info(f'Recording {n_episodes_recorded} episodes of {n_envs} car on hill copies running in parallel')

    # Agent
    agent = Agent(mdp.info, RandomPolicy(mdp.info.action_space))

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    # RUN. The recorded episodes are stored one after the other in a single video file
    core.evaluate(n_episodes=n_episodes_recorded, render=True, record=True)

    logger.info(f'{n_episodes_recorded} episodes recorded in {logger.recorded_videos[-1]}')


if __name__ == '__main__':
    experiment(n_episodes_recorded=5, n_envs=4)
