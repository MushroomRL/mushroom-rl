import numpy as np

from mushroom_rl.core import Core, MDPInfo, Box, VectorizedEnvironment, Agent
from mushroom_rl.policy import Policy
from mushroom_rl.utils.callbacks import CollectDataset


class ZeroPolicy(Policy):
    def __init__(self, action_shape):
        self._dim = action_shape[0]
        super().__init__()

    def draw_action(self, state):
        return np.zeros((state.shape[0], self._dim))


class IdentifiableVectorizedEnv(VectorizedEnvironment):
    def __init__(self):
        n_envs = 3
        observation_space = Box(0, 1000, shape=(1,))
        action_space = Box(-1, 1, shape=(1,))
        mdp_info = MDPInfo(observation_space, action_space, 0.99, 100, backend='numpy')

        self._lengths = np.array([4, 6, 8])
        self._counter = np.zeros(n_envs, dtype=int)
        self._next_id = 0
        self.terminal_ids = set()

        super().__init__(mdp_info, n_envs)

    def reset_all(self, env_mask, state=None):
        idxs = np.arange(self._n_envs)[env_mask]
        self._counter[idxs] = self._lengths[idxs]

        return self._counter.reshape(-1, 1).astype(float), [{}] * self._n_envs

    def step_all(self, env_mask, action):
        reward = -np.ones(self._n_envs)
        ids = dict()
        for e in np.arange(self._n_envs)[env_mask]:
            reward[e] = self._next_id
            ids[e] = self._next_id
            self._next_id += 1

        self._counter[env_mask] -= 1
        absorbing = self._counter <= 0

        for e in np.arange(self._n_envs)[env_mask]:
            if absorbing[e]:
                self.terminal_ids.add(ids[e])

        return self._counter.reshape(-1, 1).astype(float), reward, absorbing & env_mask, [{}] * self._n_envs


class HorizonTruncatingVectorizedEnv(VectorizedEnvironment):
    def __init__(self):
        lengths = [2, 5, 7]
        horizon = 3
        n_envs = len(lengths)
        observation_space = Box(0, 1000, shape=(1,))
        action_space = Box(-1, 1, shape=(1,))
        mdp_info = MDPInfo(observation_space, action_space, 0.99, horizon, backend='numpy')

        self._lengths = np.array(lengths)
        self._counter = np.zeros(n_envs, dtype=int)
        self._next_id = 0
        self.terminal_ids = set()

        super().__init__(mdp_info, n_envs)

    def reset_all(self, env_mask, state=None):
        idxs = np.arange(self._n_envs)[env_mask]
        self._counter[idxs] = self._lengths[idxs]

        return self._counter.reshape(-1, 1).astype(float), [{}] * self._n_envs

    def step_all(self, env_mask, action):
        reward = -np.ones(self._n_envs)
        ids = dict()
        for e in np.arange(self._n_envs)[env_mask]:
            reward[e] = self._next_id
            ids[e] = self._next_id
            self._next_id += 1

        self._counter[env_mask] -= 1
        absorbing = (self._counter <= 0) & env_mask

        for e in np.arange(self._n_envs)[env_mask]:
            if absorbing[e]:
                self.terminal_ids.add(ids[e])

        return self._counter.reshape(-1, 1).astype(float), reward, absorbing, [{}] * self._n_envs


class RecordingAgent(Agent):
    def __init__(self, mdp_info):
        policy = ZeroPolicy(mdp_info.action_space.shape)
        super().__init__(mdp_info, policy, backend='numpy')
        self.fit_rewards = list()

    def fit(self, dataset):
        self.fit_rewards += np.asarray(dataset.reward).tolist()


def test_vectorized_collection_fits_each_transition_once():
    np.random.seed(0)
    env = IdentifiableVectorizedEnv()
    agent = RecordingAgent(env.info)

    seen = list()

    def record_step(samples):
        rewards = np.asarray(samples[2])
        seen.extend(rewards[rewards >= 0].tolist())

    n_steps_per_fit = 5
    core = Core(agent, env, callback_step=record_step)
    core.learn(n_steps=200, n_steps_per_fit=n_steps_per_fit, quiet=True)

    fit = agent.fit_rewards

    assert set(fit) <= set(seen)
    assert len(fit) == len(set(fit))

    never_fit = set(seen) - set(fit)
    assert len(never_fit) < n_steps_per_fit + env._n_envs


def test_collect_dataset_collects_each_consumed_transition_once():
    np.random.seed(0)
    env = IdentifiableVectorizedEnv()
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
    real_terminals_closed = all(bool(l) for r, l in zip(collected_rewards, collected_last) if r in terminal)
    spurious = sum(1 for r, l in zip(collected_rewards, collected_last) if l and r not in terminal)

    assert real_terminals_closed
    assert spurious <= env._n_envs


def test_vectorized_collection_survives_horizon_truncation_with_episode_budget_exhaustion():
    configs = [dict(n_episodes=7, n_episodes_per_fit=3), dict(n_episodes=11, n_episodes_per_fit=4)]

    for config in configs:
        for seed in range(20):
            np.random.seed(seed)
            env = HorizonTruncatingVectorizedEnv()
            agent = RecordingAgent(env.info)

            seen = list()

            def record_step(samples):
                rewards = np.asarray(samples[2])
                seen.extend(rewards[rewards >= 0].tolist())

            core = Core(agent, env, callback_step=record_step)
            core.learn(quiet=True, **config)

            fit = agent.fit_rewards
            assert len(fit) == len(set(fit))
            assert set(fit) <= set(seen)


def test_vectorized_collection_survives_horizon_truncation_with_step_budget_exhaustion():
    configs = [dict(n_steps=50, n_steps_per_fit=5), dict(n_steps=200, n_episodes_per_fit=3)]

    for config in configs:
        for seed in range(20):
            np.random.seed(seed)
            env = HorizonTruncatingVectorizedEnv()
            agent = RecordingAgent(env.info)

            seen = list()

            def record_step(samples):
                rewards = np.asarray(samples[2])
                seen.extend(rewards[rewards >= 0].tolist())

            core = Core(agent, env, callback_step=record_step)
            core.learn(quiet=True, **config)

            fit = agent.fit_rewards
            assert len(fit) == len(set(fit))
            assert set(fit) <= set(seen)


def test_vectorized_collection_fits_each_transition_once_episodes_per_fit():
    np.random.seed(0)
    env = IdentifiableVectorizedEnv()
    agent = RecordingAgent(env.info)

    seen = list()

    def record_step(samples):
        rewards = np.asarray(samples[2])
        seen.extend(rewards[rewards >= 0].tolist())

    core = Core(agent, env, callback_step=record_step)
    core.learn(n_episodes=30, n_episodes_per_fit=3, quiet=True)

    fit = agent.fit_rewards

    assert set(fit) <= set(seen)
    assert len(fit) == len(set(fit))


def test_collect_dataset_collects_each_consumed_transition_once_episodes_per_fit():
    np.random.seed(0)
    env = IdentifiableVectorizedEnv()
    agent = RecordingAgent(env.info)
    callback = CollectDataset(initial_capacity=2)

    core = Core(agent, env, callbacks_fit=[callback])
    core.learn(n_episodes=30, n_episodes_per_fit=3, quiet=True)

    collected = callback.get()
    collected_rewards = np.asarray(collected.reward)
    collected_last = np.asarray(collected.last).astype(bool)

    assert len(collected_rewards) == len(set(collected_rewards.tolist()))
    assert sorted(collected_rewards.tolist()) == sorted(agent.fit_rewards)

    terminal = env.terminal_ids
    real_terminals_closed = all(bool(l) for r, l in zip(collected_rewards, collected_last) if r in terminal)
    spurious = sum(1 for r, l in zip(collected_rewards, collected_last) if l and r not in terminal)

    assert real_terminals_closed
    assert spurious <= env._n_envs
