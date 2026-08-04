"""
This script showcase the viewer of finite MDPs.

Each demo drives its environment with a random policy and resets whenever an episode ends, reporting the
layout the states were arranged on and the window the viewer opened for them.

    python gridworld_viewer.py                 # the environment demos, in order
    python gridworld_viewer.py --all           # also the size limit and aspect ratio ones
    python gridworld_viewer.py chain_long      # a single demo, from either group
    python gridworld_viewer.py --list          # the names of every demo
    python gridworld_viewer.py --dt .3         # slow the animation down
    python gridworld_viewer.py --steps 200     # linger on each demo

The environment demos show what each finite MDP draws:
    finite_mdp      a plain FiniteMDP with no subclass, so the default visualization: the grid and the agent,
                    a random walk over 120 states laid out on a balanced block
    grid_world      walls gray, start wheat, goal green, hole red, agent blue circle
    grid_file       the same, on the map read from examples/data/grid_world/grid.txt
    van_hasselt     after the goal reward the agent sits on the goal for one frame, then the episode restarts
    taxi            yellow passengers vanish as they are picked up: the map switches layer
    taxi_from_size  the same, on a grid built in code rather than read from a file
    chain_short     a single row of 5 cells, filling the window
    chain_long      600 states read like text: several rows, goals green
    chain_edge      the longest single row that fits the width of the screen

The demos added by ``--all`` walk up to the limits of that layout. A cell takes ``ideal_scale`` pixels, 40 by
default, and shrinks towards ``min_scale``, 20, only when the grid needs the room. Below ``min_scale`` the
grid is not drawn at all, which caps it at 1920 / 20 = 96 columns by 1080 / 20 = 54 rows:
    square_50       50x50, still drawn, with the cells already shrunk well below the ideal size
    square_54       54x54, the largest square that still fits the height of the screen
    square_55       55x55, one row too many: no viewer, render() does nothing
    widest          3x96, the last grid that fits the width of the screen
    too_wide        3x97, one column too many: no viewer
    tallest         54x3, the last grid that fits the height of the screen
    too_tall        55x3, one row too many: no viewer
    flat            a 2x20 grid, and
    tall            a 20x2 grid, the same cells rotated: a grid and its transpose are drawn at a comparable
                    cell size, each bound acting on its own axis, differing only because the screen itself
                    is not square

Every size above is written as rows x columns, the same order the demos report.

"""
import argparse

import numpy as np

from mushroom_rl.core import Logger
from mushroom_rl.environments import FiniteMDP, GridWorld, GridWorldVanHasselt, SimpleChain, Taxi
from mushroom_rl.utils.experiments import get_data_dir


def build_random_walk(n_states, prob=.9):
    """
    Build a plain FiniteMDP, a chain the agent walks along with no goal, to show the default visualization.

    """
    p = np.zeros((n_states, 2, n_states))

    for state in range(n_states):
        p[state, 0, state] += 1. - prob
        p[state, 0, min(state + 1, n_states - 1)] += prob
        p[state, 1, state] += 1. - prob
        p[state, 1, max(state - 1, 0)] += prob

    iota = np.zeros(n_states)
    iota[0] = 1.

    return FiniteMDP(p, np.zeros_like(p), iota, horizon=100)


def get_demos():
    """
    Return the environment demos, mapping each name to the environment to build and the steps to run it for.

    """
    grid_map = get_data_dir(__file__) / 'grid_world' / 'grid.txt'

    return {
        'finite_mdp': (lambda: build_random_walk(120), 40),
        'grid_world': (lambda: GridWorld.from_size(height=5, width=8, goal=(4, 7), start=(0, 0)), 40),
        'grid_file': (lambda: GridWorld.from_file(grid_map, prob=.9), 40),
        'van_hasselt': (lambda: GridWorldVanHasselt(), 40),
        'taxi': (lambda: Taxi.generate(), 80),
        'taxi_from_size': (lambda: Taxi.generate(height=5, width=5), 60),
        'chain_short': (lambda: SimpleChain(), 30),
        'chain_long': (lambda: SimpleChain(n_states=600, goal_states=[42, 300]), 40),
        'chain_edge': (lambda: SimpleChain(n_states=96, goal_states=[48]), 30),
    }


def get_edge_case_demos():
    """
    Return the demos walking up to the size limit of the layout, and the pair checking that a grid and its
    transpose are drawn at the same cell size.

    """
    return {
        'square_50': (lambda: GridWorld.from_size(height=50, width=50, goal=(49, 49)), 40),
        'square_54': (lambda: GridWorld.from_size(height=54, width=54, goal=(53, 53)), 40),
        'square_55': (lambda: GridWorld.from_size(height=55, width=55, goal=(54, 54)), 5),
        'widest': (lambda: GridWorld.from_size(height=3, width=96, goal=(2, 95)), 30),
        'too_wide': (lambda: GridWorld.from_size(height=3, width=97, goal=(2, 96)), 5),
        'tallest': (lambda: GridWorld.from_size(height=54, width=3, goal=(53, 2)), 30),
        'too_tall': (lambda: GridWorld.from_size(height=55, width=3, goal=(54, 2)), 5),
        'flat': (lambda: GridWorld.from_size(height=2, width=20, goal=(1, 19)), 30),
        'tall': (lambda: GridWorld.from_size(height=20, width=2, goal=(19, 1)), 30),
    }


def log_layout(mdp, logger):
    """
    Log the block the states were laid out on and the window the viewer opened for them, or the fact that
    the environment was too large to be drawn at all.

    This is the only place reaching into the internals of the environment and of its viewer. The row and
    column counts and the cell size are exactly what a viewer check has to look at, and none of them is
    part of the public interface.

    """
    viewer = mdp._viewer

    if viewer is None:
        logger.info('layout: too large to be drawn, no viewer')
    else:
        logger.info(f'layout: {mdp._n_rows} rows x {mdp._n_columns} columns')
        logger.info(f'window: {viewer.size[0]} x {viewer.size[1]} px, cells {viewer._scale:.1f} px')


def run_demo(name, mdp, n_steps, dt, logger):
    """
    Drive one environment with a random policy, rendering every step and resetting at the end of an episode.

    """
    mdp.info.dt = dt

    logger.weak_line()
    logger.info(f'{name}: {mdp.info.observation_space.n} states, {mdp.info.action_space.n} actions, '
                f'horizon {mdp.info.horizon}')
    log_layout(mdp, logger)

    mdp.reset()
    mdp.render()

    episodes = 0
    for _ in range(n_steps):
        _, _, absorbing, _ = mdp.step([np.random.randint(mdp.info.action_space.n)])
        mdp.render()

        if absorbing:
            episodes += 1
            mdp.reset()
            mdp.render()

    logger.info(f'ran {n_steps} steps, {episodes} episodes finished')

    mdp.stop()


def get_selected_demos(names, run_all):
    """
    Return the demos to run: the environment ones, joined by the edge case ones when they are asked for,
    and restricted to the ones named on the command line.

    """
    demos = get_demos()

    if run_all or names:
        demos |= get_edge_case_demos()

    if names:
        unknown = [name for name in names if name not in demos]
        if unknown:
            raise ValueError(f'Unknown demos {", ".join(unknown)}. Accepted demos are: {", ".join(demos)}')

        demos = {name: demos[name] for name in names}

    return demos


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('demo', nargs='*', default=[],
                        help='demos to run, by name (default: the environment ones)')
    parser.add_argument('--all', action='store_true',
                        help='also run the size limit and aspect ratio demos')
    parser.add_argument('--list', action='store_true', help='list the demos and exit')
    parser.add_argument('--dt', type=float, default=.15, help='seconds each frame stays on screen')
    parser.add_argument('--steps', type=int, default=None,
                        help='override the number of steps of every demo')
    parser.add_argument('--seed', type=int, default=0, help='seed of the random policy')

    return parser.parse_args()


def experiment(demos, dt, n_steps=None, seed=0):
    np.random.seed(seed)

    logger = Logger('gridworld_viewer', results_dir=None)
    logger.strong_line()
    logger.info(f'Finite MDP viewer check over {len(demos)} demos, dt {dt}, seed {seed}')

    for name, (build, demo_steps) in demos.items():
        run_demo(name, build(), n_steps or demo_steps, dt, logger)


if __name__ == '__main__':
    args = parse_args()

    if args.list:
        print('\n'.join(get_selected_demos([], run_all=True)))
    else:
        experiment(get_selected_demos(args.demo, args.all), args.dt, args.steps, args.seed)
