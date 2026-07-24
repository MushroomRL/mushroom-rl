import numpy as np

from mushroom_rl.core import Core, MDPInfo, Box, Environment, Agent
from mushroom_rl.policy import Policy
from mushroom_rl.utils.callbacks import CollectDataset


class ZeroPolicy(Policy):
    def __init__(self, action_shape):
        self._dim = action_shape[0]
        super().__init__()

    def draw_action(self, state):
        return np.zeros(self._dim)


class IdentifiableEnv(Environment):
    def __init__(self):
        observation_space = Box(0, 1000, shape=(1,))
        action_space = Box(-1, 1, shape=(1,))
        mdp_info = MDPInfo(observation_space, action_space, 0.99, 100, backend='numpy')

        self._lengths = [4, 6, 8]
        self._next_length = 0
        self._counter = 0
        self._next_id = 0
        self.terminal_ids = set()

        super().__init__(mdp_info)

    def reset(self, state=None):
        self._counter = self._lengths[self._next_length % len(self._lengths)]
        self._next_length += 1

        return np.array([float(self._counter)]), {}

    def step(self, action):
        current_id = self._next_id
        reward = float(current_id)
        self._next_id += 1

        self._counter -= 1
        absorbing = self._counter <= 0

        if absorbing:
            self.terminal_ids.add(current_id)

        return np.array([float(self._counter)]), reward, absorbing, {}


class RecordingAgent(Agent):
    def __init__(self, mdp_info):
        policy = ZeroPolicy(mdp_info.action_space.shape)
        super().__init__(mdp_info, policy, backend='numpy')
        self.fit_rewards = list()

    def fit(self, dataset):
        self.fit_rewards += np.asarray(dataset.reward).tolist()


def test_sequential_collection_fits_each_transition_once():
    np.random.seed(0)
    env = IdentifiableEnv()
    agent = RecordingAgent(env.info)

    seen = list()

    def record_step(sample):
        seen.append(sample[2])

    core = Core(agent, env, callback_step=record_step)
    core.learn(n_steps=200, n_steps_per_fit=5, quiet=True)

    fit = agent.fit_rewards

    assert sorted(fit) == sorted(seen)
    assert len(fit) == len(set(fit))


def test_collect_dataset_collects_each_consumed_transition_once_sequential():
    np.random.seed(0)
    env = IdentifiableEnv()
    agent = RecordingAgent(env.info)
    callback = CollectDataset(initial_capacity=2)

    core = Core(agent, env, callbacks_fit=[callback])
    core.learn(n_steps=200, n_steps_per_fit=5, quiet=True)

    collected = callback.get()
    collected_rewards = np.asarray(collected.reward)
    collected_last = np.asarray(collected.last).astype(bool)

    assert len(collected_rewards) == len(set(collected_rewards.tolist()))
    assert sorted(collected_rewards.tolist()) == sorted(agent.fit_rewards)

    terminal = env.terminal_ids
    for r, l in zip(collected_rewards, collected_last):
        assert bool(l) == (r in terminal)


def test_sequential_collection_fits_each_transition_once_episodes_per_fit():
    np.random.seed(0)
    env = IdentifiableEnv()
    agent = RecordingAgent(env.info)

    seen = list()

    def record_step(sample):
        seen.append(sample[2])

    core = Core(agent, env, callback_step=record_step)
    core.learn(n_episodes=32, n_episodes_per_fit=4, quiet=True)

    fit = agent.fit_rewards

    assert sorted(fit) == sorted(seen)
    assert len(fit) == len(set(fit))


def test_collect_dataset_collects_each_consumed_transition_once_sequential_episodes_per_fit():
    np.random.seed(0)
    env = IdentifiableEnv()
    agent = RecordingAgent(env.info)
    callback = CollectDataset(initial_capacity=2)

    core = Core(agent, env, callbacks_fit=[callback])
    core.learn(n_episodes=32, n_episodes_per_fit=4, quiet=True)

    collected = callback.get()
    collected_rewards = np.asarray(collected.reward)
    collected_last = np.asarray(collected.last).astype(bool)

    assert len(collected_rewards) == len(set(collected_rewards.tolist()))
    assert sorted(collected_rewards.tolist()) == sorted(agent.fit_rewards)

    terminal = env.terminal_ids
    for r, l in zip(collected_rewards, collected_last):
        assert bool(l) == (r in terminal)
