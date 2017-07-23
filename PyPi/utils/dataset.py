import numpy as np


def parse_dataset(dataset):
    """
    Split the dataset in its different components and return them.

    # Arguments
        dataset (list): the dataset to parse.

    # Returns
        The np.array of state, action, reward, next_state, absorbing flag and
        last step flag.
    """
    assert len(dataset) > 0

    state = list()
    action = list()
    reward = list()
    next_state = list()
    absorbing = list()
    last = list()

    for i in xrange(len(dataset)):
        state.append(dataset[i][0])
        action.append(dataset[i][1])
        reward.append(dataset[i][2])
        next_state.append(dataset[i][3])
        absorbing.append(dataset[i][4])
        last.append(dataset[i][5])

    return np.array(state), np.array(action), np.array(reward), np.array(
        next_state), np.array(absorbing), np.array(last)


def select_episodes(dataset, state_dim, action_dim, n_episodes, parse=False):
    """
    Return the desired number of episodes in the provided dataset.

    # Arguments
        dataset (np.array): the dataset to parse;
        state_dim (int > 0): the dimension of the MDP state;
        action_dim (int > 0): the dimension of the MDP action;
        n_episodes (int >= 0): the number of episodes to pick from the dataset;
        parse (bool): whether to parse the dataset to return.

    # Returns
        A subset of the dataset containing the desired number of episodes.
    """
    assert n_episodes >= 0, 'Number of episodes must be greater than or equal' \
                            'to zero.'
    if n_episodes == 0:
        return np.array([[]])

    dataset = np.array(dataset)
    last_idxs = np.argwhere(dataset[:, -1] == 1).ravel()
    sub_dataset = dataset[:last_idxs[n_episodes - 1] + 1, :]

    return sub_dataset if not parse else parse_dataset(sub_dataset, state_dim,
                                                       action_dim)


def select_samples(dataset, n_samples, parse=False):
    """
    Return the desired number of samples in the provided dataset.

    # Arguments
        dataset (np.array): the dataset to parse;
        n_episodes (int >= 0): the number of samples to pick from the dataset;
        parse (bool): whether to parse the dataset to return.

    # Returns
        A subset of the dataset containing the desired number of samples.
    """
    assert n_samples >= 0, 'Number of samples must be greater than or equal' \
                           'to zero.'
    if n_samples == 0:
        return np.array([[]])

    dataset = np.array(dataset)
    idxs = np.random.randint(dataset.shape[0], size=n_samples)
    sub_dataset = dataset[idxs, ...]
    return sub_dataset if not parse else parse_dataset(sub_dataset)


def compute_J(dataset, gamma=1.):
    """
    Compute the J.

    Arguments
        dataset (list): the dataset to consider to compute J;
        gamma (float): discount factor.

    Returns
        the average cumulative discounted reward.
    """
    _, _, reward, _, _, last = parse_dataset(dataset)
    js = list()

    j = 0.
    episode_steps = 0
    for i in xrange(reward.size):
        j += gamma ** episode_steps * reward[i]
        episode_steps += 1
        if last[i]:
            js.append(j)
            j = 0.
            episode_steps = 0

    return js


def max_QA(states, absorbing, approximator, actions):
    """
    # Arguments
        state (np.array): the state where the agent is;
        absorbing (np.array): whether the state is absorbing or not;
        approximator (object): the approximator to use to compute the
            action values;
        actions (np.array): the values of the discrete actions.

    # Returns
        A np.array of maximum action values and a np.array of their
        corresponding actions.
    """
    if states.ndim == 1:
        states = np.expand_dims(states, axis=0)

    q = approximator.predict_all(states, actions)
    if np.any(absorbing):
        q *= 1 - absorbing.reshape(-1, 1)

    max_q = np.max(q, axis=1)
    if q.shape[0] > 1:
        max_a = np.argmax(q, axis=1)
    else:
        max_a = [np.random.choice(np.argwhere(q[0] == max_q).ravel())]

    return max_q, actions[max_a]


def state_action_idx(state, action):
    """
    Concatenate state and action samples in a single tuple. This can be used,
    for instance, when using the state-action tuple as an index. 'state' and
    'action' must have the same shape.

    Arguments
        state (np.array): the array of states with shape (n, state_dim);
        action (np.array): the array of actions with shape (n, action_dim).

    Returns
        the tuple of the concatenation of the 'state' and 'action' array with
        shape (n, state_dim + action_dim).
    """
    if state.ndim == 2 and action.ndim == 2:
        return tuple(np.concatenate((state, action), axis=1).ravel())
    else:
        raise ValueError('Wrong dimensionality.')
