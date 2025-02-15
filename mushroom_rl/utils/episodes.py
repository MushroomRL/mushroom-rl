from mushroom_rl.core.array_backend import ArrayBackend

def split_episodes(last, *arrays):
    """
    Split a array from shape (n_steps) to (n_episodes, max_episode_steps).
    """
    backend = ArrayBackend.get_array_backend_from(last)
    
    if last.sum().item() <= 1:
        return arrays if len(arrays) > 1 else arrays[0]

    row_idx, colum_idx, n_episodes, max_episode_steps = _get_episode_idx(last, backend)
    episodes_arrays = []

    for array in arrays:
        array_ep = backend.zeros(n_episodes, max_episode_steps, *array.shape[1:], dtype=array.dtype, device=array.device if backend.get_backend_name() == "torch" else None)

        array_ep[row_idx, colum_idx] = array
        episodes_arrays.append(array_ep)

    return episodes_arrays if len(episodes_arrays) > 1 else episodes_arrays[0]

def unsplit_episodes(last, *episodes_arrays):
    """
    Unsplit a array from shape (n_episodes, max_episode_steps) to (n_steps).
    """
    
    if last.sum().item() <= 1:
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

    last = backend.copy(last)
    last[-1] = True

    n_episodes = last.sum()
    last_idx = backend.nonzero(last).squeeze()
    first_steps = backend.from_list([last_idx[0] + 1])
    if backend.get_backend_name() == 'torch':
        first_steps = first_steps.to(last.device)
    episode_steps = backend.concatenate([first_steps, last_idx[1:] - last_idx[:-1]])
    max_episode_steps = episode_steps.max()

    start_idx = backend.concatenate([backend.zeros(1, dtype=int, device=last.device if backend.get_backend_name() == 'torch' else None), last_idx[:-1] + 1])
    range_n_episodes = backend.arange(0, n_episodes, dtype=int)
    range_len = backend.arange(0, last.shape[0], dtype=int)
    if backend.get_backend_name() == 'torch':
        range_n_episodes = range_n_episodes.to(last.device)
        range_len = range_len.to(last.device)
    row_idx = backend.repeat(range_n_episodes, episode_steps)
    colum_idx = range_len - start_idx[row_idx]

    return row_idx, colum_idx, n_episodes, max_episode_steps