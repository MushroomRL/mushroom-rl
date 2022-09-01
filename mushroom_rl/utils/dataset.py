import numpy as np

from mushroom_rl.utils.frames import LazyFrames


def parse_dataset(dataset, features=None):
    """
    Split the dataset in its different components and return them.

    Args:
        dataset (list): the dataset to parse;
        features (object, None): features to apply to the states.

    Returns:
        The np.ndarray of state, action, reward, next_state, absorbing flag and
        last step flag. Features are applied to ``state`` and ``next_state``,
        when provided.

    """
    assert len(dataset) > 0

    shape = dataset[0][0].shape if features is None else (features.size,)

    state = np.ones((len(dataset),) + shape)
    action = np.ones((len(dataset),) + dataset[0][1].shape)
    reward = np.ones(len(dataset))
    next_state = np.ones((len(dataset),) + shape)
    absorbing = np.ones(len(dataset))
    last = np.ones(len(dataset))

    if features is not None:
        for i in range(len(dataset)):
            state[i, ...] = features(dataset[i][0])
            action[i, ...] = dataset[i][1]
            reward[i] = dataset[i][2]
            next_state[i, ...] = features(dataset[i][3])
            absorbing[i] = dataset[i][4]
            last[i] = dataset[i][5]
    else:
        for i in range(len(dataset)):
            state[i, ...] = dataset[i][0]
            action[i, ...] = dataset[i][1]
            reward[i] = dataset[i][2]
            next_state[i, ...] = dataset[i][3]
            absorbing[i] = dataset[i][4]
            last[i] = dataset[i][5]

    return np.array(state), np.array(action), np.array(reward), np.array(
        next_state), np.array(absorbing), np.array(last)


def arrays_as_dataset(states, actions, rewards, next_states, absorbings, lasts):
    """
    Creates a dataset of transitions from the provided arrays.

    Args:
        states (np.ndarray): array of states;
        actions (np.ndarray): array of actions;
        rewards (np.ndarray): array of rewards;
        next_states (np.ndarray): array of next_states;
        absorbings (np.ndarray): array of absorbing flags;
        lasts (np.ndarray): array of last flags.

    Returns:
        The list of transitions.

    """
    assert (len(states) == len(actions) == len(rewards)
            == len(next_states) == len(absorbings) == len(lasts))

    dataset = list()
    for s, a, r, ss, ab, last in zip(states, actions, rewards, next_states,
                                     absorbings.astype(bool), lasts.astype(bool)
                                     ):
        dataset.append((s, a, r.item(0), ss, ab.item(0), last.item(0)))

    return dataset


def compute_episodes_length(dataset):
    """
    Compute the length of each episode in the dataset.

    Args:
        dataset (list): the dataset to consider.

    Returns:
        A list of length of each episode in the dataset.

    """
    lengths = list()
    l = 0
    for sample in dataset:
        l += 1
        if sample[-1] == 1:
            lengths.append(l)
            l = 0

    return lengths


def select_first_episodes(dataset, n_episodes, parse=False):
    """
    Return the first ``n_episodes`` episodes in the provided dataset.

    Args:
        dataset (list): the dataset to consider;
        n_episodes (int): the number of episodes to pick from the dataset;
        parse (bool, False): whether to parse the dataset to return.

    Returns:
        A subset of the dataset containing the first ``n_episodes`` episodes.

    """
    assert n_episodes >= 0, 'Number of episodes must be greater than or equal' \
                            'to zero.'
    if n_episodes == 0:
        return np.array([[]])

    dataset = np.array(dataset, dtype=object)
    last_idxs = np.argwhere(dataset[:, -1] == 1).ravel()
    sub_dataset = dataset[:last_idxs[n_episodes - 1] + 1, :]

    return sub_dataset if not parse else parse_dataset(sub_dataset)


def select_random_samples(dataset, n_samples, parse=False):
    """
    Return the randomly picked desired number of samples in the provided
    dataset.

    Args:
        dataset (list): the dataset to consider;
        n_samples (int): the number of samples to pick from the dataset;
        parse (bool, False): whether to parse the dataset to return.

    Returns:
        A subset of the dataset containing randomly picked ``n_samples``
        samples.

    """
    assert n_samples >= 0, 'Number of samples must be greater than or equal' \
                           'to zero.'
    if n_samples == 0:
        return np.array([[]])

    dataset = np.array(dataset, dtype=object)
    idxs = np.random.randint(dataset.shape[0], size=n_samples)
    sub_dataset = dataset[idxs, ...]

    return sub_dataset if not parse else parse_dataset(sub_dataset)


def get_init_states(dataset):
    """
    Get the initial states of a dataset

    Args:
        dataset (list): the dataset to consider.

    Returns:
        An array of initial states of the considered dataset.

    """
    pick = True
    x_0 = list()
    for d in dataset:
        if pick:
            if isinstance(d[0], LazyFrames):
                x_0.append(np.array(d[0]))
            else:
                x_0.append(d[0])
        pick = d[-1]
    return np.array(x_0)


def compute_J(dataset, gamma=1.):
    """
    Compute the cumulative discounted reward of each episode in the dataset.

    Args:
        dataset (list): the dataset to consider;
        gamma (float, 1.): discount factor.

    Returns:
        The cumulative discounted reward of each episode in the dataset.

    """
    js = list()

    j = 0.
    episode_steps = 0
    for i in range(len(dataset)):
        j += gamma ** episode_steps * dataset[i][2]
        episode_steps += 1
        if dataset[i][-1] or i == len(dataset) - 1:
            js.append(j)
            j = 0.
            episode_steps = 0

    if len(js) == 0:
        return [0.]
    return js


def compute_metrics(dataset, gamma=1.):
    """
    Compute the metrics of each complete episode in the dataset.

    Args:
        dataset (list): the dataset to consider;
        gamma (float, 1.): the discount factor.

    Returns:
        The minimum score reached in an episode,
        the maximum score reached in an episode,
        the mean score reached,
        the median score reached,
        the number of completed episodes.

        If no episode has been completed, it returns 0 for all values.

    """
    for i in reversed(range(len(dataset))):
        if dataset[i][-1]:
            i += 1
            break

    dataset = dataset[:i]

    if len(dataset) > 0:
        J = compute_J(dataset, gamma)
        return np.min(J), np.max(J), np.mean(J), np.median(J), len(J)
    else:
        return 0, 0, 0, 0, 0
