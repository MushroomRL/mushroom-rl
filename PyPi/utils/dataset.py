import numpy as np


def parse_dataset(dataset, state_dim, action_dim):
    """
    Split the dataset in its different components and return them.

    # Arguments
        dataset (np.array): the dataset to parse.
        state_dim (int > 0): the dimension of the MDP state.
        action_dim (int > 0): the dimension of the MDP action.

    # Returns
        The np.array of state, action, reward, next_state, absorbing flag and
        last step flag.
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


def select_episodes(dataset, state_dim, action_dim, n_episodes, parse=False):
    """
    Return the desired number of episodes in the provided dataset.

    # Arguments
        dataset (np.array): the dataset to parse.
        state_dim (int > 0): the dimension of the MDP state.
        action_dim (int > 0): the dimension of the MDP action.
        n_episodes (int >= 0): the number of episodes to pick from the dataset.
        parse (bool): whether to parse the dataset to return.

    # Returns
        A subset of the dataset containing the desired number of episodes.
    """
    assert n_episodes >= 0, 'Number of episodes must be greater than or equal' \
                            'to zero.'
    if n_episodes == 0:
        return np.array([[]])

    last_idxs = np.argwhere(dataset[:, -1] == 1).ravel()
    sub_dataset = dataset[:last_idxs[n_episodes - 1] + 1, :]

    return sub_dataset if not parse else parse_dataset(sub_dataset, state_dim,
                                                       action_dim)
