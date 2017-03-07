import numpy as np


def parse_dataset(dataset, state_dim, action_dim):
    if isinstance(dataset, list):
        dataset = np.array(dataset)
    if len(dataset.shape) == 1:
        dataset = np.expand_dims(dataset, 0)

    reward_idx = state_dim + action_dim

    state = dataset[:, :state_dim]
    action = dataset[:, state_dim:reward_idx]
    reward = dataset[:, reward_idx]
    next_state = dataset[:, reward_idx + 1:reward_idx + 1 + state_dim]
    absorbing = dataset[:, -2]
    last = dataset[:, -1]

    return state, action, reward, next_state, absorbing, last


def select_episodes(dataset, state_dim, action_dim, parse=False):
    if parse:
        dataset = parse_dataset(dataset, state_dim, action_dim)

    pass
