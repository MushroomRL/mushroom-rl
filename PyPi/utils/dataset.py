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

    dataset = np.array(dataset)
    last_idxs = np.argwhere(dataset[:, -1] == 1).ravel()
    sub_dataset = dataset[:last_idxs[n_episodes - 1] + 1, :]

    return sub_dataset if not parse else parse_dataset(sub_dataset, state_dim,
                                                       action_dim)


def select_samples(dataset, state_dim, action_dim, n_samples, parse=False):
    """
    Return the desired number of samples in the provided dataset.

    # Arguments
        dataset (np.array): the dataset to parse.
        state_dim (int > 0): the dimension of the MDP state.
        action_dim (int > 0): the dimension of the MDP action.
        n_episodes (int >= 0): the number of samples to pick from the dataset.
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


def max_QA(states, absorbing, approximator, discrete_actions):
    """
    # Arguments
        state (np.array): the state where the agent is.
        absorbing (np.array): whether the state is absorbing or not.
        approximator (object): the approximator to use to compute the
            action values.
        discrete_actions (np.array): the values of the discrete actions.

    # Returns
        A np.array of maximum action values and a np.array of their
        corresponding actions.
    """
    if states.ndim == 1:
        states = np.expand_dims(states, axis=0)

    n_states = states.shape[0]
    n_actions = discrete_actions.shape[0]
    action_dim = discrete_actions.shape[1]

    Q = np.zeros((n_states, n_actions))
    for action in xrange(n_actions):
        actions = np.repeat(discrete_actions[action],
                            n_states,
                            0).reshape(-1, 1)

        samples = [states, actions]

        predictions = approximator.predict(samples)

        Q[:, action] = predictions * (1 - absorbing)

    if Q.shape[0] > 1:
        amax = np.argmax(Q, axis=1)
    else:
        q = Q[0]
        amax = [np.random.choice(np.argwhere(q == np.max(q)).ravel())]

    # store Q-value and action for each state
    r_q, r_a = np.zeros(n_states), np.zeros((n_states, action_dim))
    for idx in xrange(n_states):
        r_q[idx] = Q[idx, amax[idx]]
        r_a[idx] = discrete_actions[amax[idx]]

    return r_q, r_a
