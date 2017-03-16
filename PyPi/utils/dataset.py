import numpy as np


def parse_dataset(dataset, state_dim, action_dim):
    """
    Splits the dataset in its different components and return them.

    # Arguments
        dataset (np.array): the dataset to parse.
        state_dim (int > 0): the dimension of the MDP state.
        action_dim (int > 0): the dimension of the MDP action.

    # Returns
        the state, action, reward, next_state, absorbing flag and last step flag
        arrays.
    """
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
