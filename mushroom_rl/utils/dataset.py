import numpy as np

from mushroom_rl.utils.frames import LazyFrames


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


