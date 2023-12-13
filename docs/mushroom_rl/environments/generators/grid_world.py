import numpy as np

from mushroom_rl.environments.finite_mdp import FiniteMDP


def generate_grid_world(grid, prob, pos_rew, neg_rew, gamma=.9, horizon=100):
    """
    This Grid World generator requires a .txt file to specify the
    shape of the grid world and the cells. There are five types of cells: 'S' is
    the starting position where the agent is; 'G' is the goal state; '.' is a
    normal cell; '*' is a hole, when the agent steps on a hole, it receives a
    negative reward and the episode ends; '#' is a wall, when the agent is
    supposed to step on a wall, it actually remains in its current state. The
    initial states distribution is uniform among all the initial states
    provided.

    The grid is expected to be rectangular.

    Args:
        grid (str): the path of the file containing the grid structure;
        prob (float): probability of success of an action;
        pos_rew (float): reward obtained in goal states;
        neg_rew (float): reward obtained in "hole" states;
        gamma (float, .9): discount factor;
        horizon (int, 100): the horizon.

    Returns:
        A FiniteMDP object built with the provided parameters.

    """
    grid_map, cell_list = parse_grid(grid)
    p = compute_probabilities(grid_map, cell_list, prob)
    r = compute_reward(grid_map, cell_list, pos_rew, neg_rew)
    mu = compute_mu(grid_map, cell_list)

    return FiniteMDP(p, r, mu, gamma, horizon)


def parse_grid(grid):
    """
    Parse the grid file:

    Args:
        grid (str): the path of the file containing the grid structure;

    Returns:
        A list containing the grid structure.

    """
    grid_map = list()
    cell_list = list()
    with open(grid, 'r') as f:
        m = f.read()

        assert 'S' in m and 'G' in m

        row = list()
        row_idx = 0
        col_idx = 0
        for c in m:
            if c in ['#', '.', 'S', 'G', '*']:
                row.append(c)
                if c in ['.', 'S', 'G', '*']:
                    cell_list.append([row_idx, col_idx])
                col_idx += 1
            elif c == '\n':
                grid_map.append(row)
                row = list()
                row_idx += 1
                col_idx = 0
            else:
                raise ValueError('Unknown marker.')

    return grid_map, cell_list


def compute_probabilities(grid_map, cell_list, prob):
    """
    Compute the transition probability matrix.

    Args:
        grid_map (list): list containing the grid structure;
        cell_list (list): list of non-wall cells;
        prob (float): probability of success of an action.

    Returns:
        The transition probability matrix;

    """
    g = np.array(grid_map)
    c = np.array(cell_list)
    n_states = len(cell_list)
    p = np.zeros((n_states, 4, n_states))
    directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]

    for i in range(len(c)):
        state = c[i]

        if g[tuple(state)] in ['.', 'S']:
            for a in range(len(directions)):
                new_state = state + directions[a]
                j = np.where((c == new_state).all(axis=1))[0]
                if j.size > 0:
                    assert j.size == 1

                    p[i, a, i] = 1. - prob
                    p[i, a, j] = prob
                else:
                    p[i, a, i] = 1.

    return p


def compute_reward(grid_map, cell_list, pos_rew, neg_rew):
    """
    Compute the reward matrix.

    Args:
        grid_map (list): list containing the grid structure;
        cell_list (list): list of non-wall cells;
        pos_rew (float): reward obtained in goal states;
        neg_rew (float): reward obtained in "hole" states;

    Returns:
        The reward matrix.

    """
    g = np.array(grid_map)
    c = np.array(cell_list)
    n_states = len(c)
    r = np.zeros((n_states, 4, n_states))
    directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]

    def give_reward(t, rew):
        for x in np.argwhere(g == t):
            j = np.where((c == x).all(axis=1))[0]

            for a in range(len(directions)):
                prev_state = x - directions[a]
                if prev_state.tolist() in c.tolist():
                    i = np.where((c == prev_state).all(axis=1))[0]
                    r[i, a, j] = rew

    give_reward('G', pos_rew)
    give_reward('*', neg_rew)

    return r


def compute_mu(grid_map, cell_list):
    """
    Compute the initial states distribution.

    Args:
        grid_map (list): list containing the grid structure;
        cell_list (list): list of non-wall cells.

    Returns:
        The initial states distribution.

    """
    g = np.array(grid_map)
    c = np.array(cell_list)
    n_states = len(c)
    mu = np.zeros(n_states)
    starts = np.argwhere(g == 'S')

    for s in starts:
        i = np.where((c == s).all(axis=1))[0]
        mu[i] = 1. / len(starts)

    return mu
