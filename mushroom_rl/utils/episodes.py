from mushroom_rl.core.array_backend import ArrayBackend


def split_episodes(last, *arrays):
    """
    Split flat arrays into per-episode ones, zero-padding the episodes shorter than the longest.

    Args:
        last: the episode-end flags;
        *arrays: the arrays to split.

    Returns:
        The episode-end flags marking the final step of every episode, including a truncated last one,
        followed by each of the split arrays.

    """
    backend = ArrayBackend.get_array_backend_from(last)

    last = backend.copy(last)
    last[-1] = True

    if last.sum().item() <= 1:
        return (last, *arrays)

    row_idx, colum_idx, n_episodes, max_episode_steps = _get_episode_idx(last, backend)
    episodes_arrays = []

    for array in (last, *arrays):
        device = backend.get_device(array)
        array_ep = backend.zeros(n_episodes, max_episode_steps, *array.shape[1:], dtype=array.dtype, device=device)

        array_ep[row_idx, colum_idx] = array
        episodes_arrays.append(array_ep)

    return tuple(episodes_arrays)


def unsplit_episodes(last, *episodes_arrays):
    """
    Concatenate per-episode arrays back into flat ones, dropping the padding of the shorter episodes.

    Args:
        last: the episode-end flags of the flat arrays;
        *episodes_arrays: the arrays to unsplit, of shape ``(n_episodes, max_episode_steps, ...)``.

    Returns:
        Each unsplit array, of shape ``(n_steps, ...)``, or the array itself if a single one was given.

    """

    if last[:-1].sum().item() == 0:
        return episodes_arrays if len(episodes_arrays) > 1 else episodes_arrays[0]

    row_idx, colum_idx, _, _ = _get_episode_idx(last)
    arrays = []

    for episode_array in episodes_arrays:
        array = episode_array[row_idx, colum_idx]
        arrays.append(array)

    return arrays if len(arrays) > 1 else arrays[0]


def _get_episode_idx(last, backend=None):
    if backend is None:
        backend = ArrayBackend.get_array_backend_from(last)

    device = backend.get_device(last)

    last = backend.copy(last)
    last[-1] = True

    n_episodes = last.sum()
    last_idx = backend.nonzero(last).squeeze()
    first_steps = backend.from_list([last_idx[0] + 1], device=device)
    episode_steps = backend.concatenate([first_steps, last_idx[1:] - last_idx[:-1]])
    max_episode_steps = episode_steps.max()

    start_idx = backend.concatenate([backend.zeros(1, dtype=int, device=device), last_idx[:-1] + 1])
    range_n_episodes = backend.arange(0, n_episodes, dtype=int, device=device)
    range_len = backend.arange(0, last.shape[0], dtype=int, device=device)
    row_idx = backend.repeat(range_n_episodes, episode_steps)
    colum_idx = range_len - start_idx[row_idx]

    return row_idx, colum_idx, n_episodes, max_episode_steps
