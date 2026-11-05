import numpy as np
import pytest

from helper.dummies import CountingVecEnv, RecordingAgent

from mushroom_rl.core import Core, MDPInfo, Box, Environment, VectorizedEnvironment, VectorizedDataset
from mushroom_rl.utils.callbacks import CollectDataset


class IdentifiableEnv(Environment):
    def __init__(self):
        mdp_info = MDPInfo(Box(0, 1000, shape=(1,)), Box(-1, 1, shape=(1,)), 0.99, 100, backend='numpy')

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
        self._next_id += 1

        self._counter -= 1
        absorbing = self._counter <= 0

        if absorbing:
            self.terminal_ids.add(current_id)

        return np.array([float(self._counter)]), float(current_id), absorbing, {}


class IdentifiableVecEnv(VectorizedEnvironment):
    def __init__(self, lengths=(4, 6, 8), horizon=100):
        mdp_info = MDPInfo(Box(0, 1000, shape=(1,)), Box(-1, 1, shape=(1,)), 0.99, horizon, backend='numpy')

        self._lengths = np.array(lengths)
        self._counter = np.zeros(len(lengths), dtype=int)
        self._next_id = 0
        self.terminal_ids = set()

        super().__init__(mdp_info, len(lengths))

    def reset_all(self, env_mask, state=None):
        self._counter[env_mask] = self._lengths[env_mask]

        return self._counter.reshape(-1, 1).astype(float), [{}] * self._n_envs

    def step_all(self, env_mask, action):
        reward = -np.ones(self._n_envs)
        for e in np.arange(self._n_envs)[env_mask]:
            reward[e] = self._next_id
            self._next_id += 1

        self._counter[env_mask] -= 1
        absorbing = (self._counter <= 0) & env_mask
        self.terminal_ids |= set(reward[absorbing].tolist())

        return self._counter.reshape(-1, 1).astype(float), reward, absorbing, [{}] * self._n_envs


def run(env, **budget):
    agent = RecordingAgent(env.info)
    seen = list()
    collector = CollectDataset(initial_capacity=2)

    def record_step(samples):
        rewards = np.atleast_1d(samples[2])
        seen.extend(rewards[rewards >= 0].tolist())

    Core(agent, env, callback_step=record_step, callbacks_fit=[collector]).learn(quiet=True, **budget)

    return agent.fit_rewards, seen, collector.get()


@pytest.mark.parametrize('budget', [dict(n_steps=200, n_steps_per_fit=5), dict(n_episodes=32, n_episodes_per_fit=4)])
def test_sequential(budget):
    env = IdentifiableEnv()
    fit, seen, collected = run(env, **budget)
    rewards = np.asarray(collected.reward).tolist()
    lasts = np.asarray(collected.last).astype(bool).tolist()

    assert sorted(fit) == sorted(seen)
    assert len(fit) == len(set(fit))
    assert sorted(rewards) == sorted(fit)
    assert lasts == [reward in env.terminal_ids for reward in rewards]


@pytest.mark.parametrize('budget', [dict(n_steps=200, n_steps_per_fit=5), dict(n_episodes=30, n_episodes_per_fit=3)])
def test_vectorized(budget):
    env = IdentifiableVecEnv()
    fit, seen, collected = run(env, **budget)
    rewards = np.asarray(collected.reward).tolist()
    lasts = np.asarray(collected.last).astype(bool).tolist()

    assert set(fit) <= set(seen)
    assert len(fit) == len(set(fit))
    assert sorted(rewards) == sorted(fit)
    assert all(last for reward, last in zip(rewards, lasts) if reward in env.terminal_ids)
    assert sum(last and reward not in env.terminal_ids for reward, last in zip(rewards, lasts)) <= env.number
    if 'n_steps' in budget:
        assert len(set(seen) - set(fit)) < budget['n_steps_per_fit'] + env.number


@pytest.mark.parametrize('budget', [dict(n_episodes=7, n_episodes_per_fit=3), dict(n_episodes=11, n_episodes_per_fit=4),
                                    dict(n_steps=50, n_steps_per_fit=5), dict(n_steps=200, n_episodes_per_fit=3)])
def test_horizon_truncation(budget):
    fit, seen, _ = run(IdentifiableVecEnv(lengths=(2, 5, 7), horizon=3), **budget)

    assert len(fit) == len(set(fit))
    assert set(fit) <= set(seen)


def test_fit_callbacks():
    env = CountingVecEnv(3, horizon=10)
    kinds = list()
    collector = CollectDataset(initial_capacity=8)
    core = Core(RecordingAgent(env.info), env, callbacks_fit=[lambda dataset: kinds.append(type(dataset)), collector])

    core.learn(n_steps=60, n_steps_per_fit=30, quiet=True)
    collected = collector.get()

    assert kinds == [VectorizedDataset, VectorizedDataset]
    assert len(collected) == 60
    assert collected.n_episodes == 6
    assert np.array_equal(collected.episodes_length, np.full(6, 10))
