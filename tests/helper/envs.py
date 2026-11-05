import numpy as np


def discrete_action(mdp):
    return [np.random.randint(mdp.info.action_space.n)]


def box_action(mdp):
    return [np.random.rand(mdp.info.action_space.shape[0])]


def scalar_action(mdp):
    return [np.random.rand()]


def rollout(mdp, action, n_steps=10, render=False):
    mdp.reset()
    if render:
        mdp.render()

    states, snapshots = list(), list()
    for _ in range(n_steps):
        state, _, _, _ = mdp.step(action(mdp))
        states.append(state)
        snapshots.append(np.copy(state))

    for state, snapshot in zip(states, snapshots):
        assert np.allclose(state, snapshot)

    if render:
        mdp.render()

    return state


def constant_rollout(mdp, action, n_steps=20):
    mdp.reset()
    observations = list()
    for _ in range(n_steps):
        observation, _, _, _ = mdp.step(action)
        assert len(observation) == len(mdp.info.observation_space.low)
        assert len(observation) == len(mdp.info.observation_space.high)
        observations.append(observation)

    return observations
