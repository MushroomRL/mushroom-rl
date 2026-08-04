"""
This script aims to replicate the experiments on the Grid World MDP as presented in:
"Double Q-Learning", Hasselt H. V. 2010.

SARSA and many variants of Q-Learning are used. The figure is drawn by ``plot_results``, which takes the
curves rather than computing them, so that it can also be imported and called on the arrays logged by a run
that already happened.

"""
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

from mushroom_rl.algorithms.value import QLearning, DoubleQLearning, \
    WeightedQLearning, SpeedyQLearning, SARSA
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import GridWorldVanHasselt
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.callbacks import CollectDataset, CollectMaxQ
from mushroom_rl.rl_utils.parameters import DecayParameter
from mushroom_rl.utils.experiments import get_log_dir

matplotlib.use('Agg')


def plot_results(curves, exp, window):
    """
    Draw the average reward per step and the max action value at the start state, one line per algorithm.
    The reward curve is smoothed with a moving average, so it is shorter than the max Q one and is drawn
    against the steps its window ends on.

    Args:
        curves (dict): mapping from algorithm name to its reward and max Q curves;
        exp (float): the decay exponent of the learning rate the curves were collected with;
        window (int): the width of the moving average the reward curve was smoothed with.

    Returns:
        The figure holding the two plots.

    """
    fig = plt.figure()
    fig.suptitle(f'Learning rate decaying as 1 / n^{exp}')

    ax_r = fig.add_subplot(2, 1, 1)
    ax_q = fig.add_subplot(2, 1, 2)

    for name, (r, max_Qs) in curves.items():
        ax_r.plot(np.arange(window, len(max_Qs) + 1), r, label=name)
        ax_q.plot(np.arange(1, len(max_Qs) + 1), max_Qs, label=name)

    ax_r.set_ylabel(f'reward per step (over {window} steps)')
    ax_q.set_ylabel('max Q at the start state')
    ax_q.set_xlabel('steps')

    for ax in (ax_r, ax_q):
        ax.legend()

    return fig


def experiment(algorithm_class, exp, seed):
    np.random.seed(seed)

    # MDP
    mdp = GridWorldVanHasselt()

    # Policy
    epsilon = DecayParameter(value=1, exp=.5, shape=mdp.info.observation_space.size)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = DecayParameter(value=1, exp=exp, shape=mdp.info.size)
    agent = algorithm_class(mdp.info, pi, learning_rate=learning_rate)

    # Algorithm
    start = np.argwhere(mdp.iota > 0).ravel()
    collect_max_Q = CollectMaxQ(agent.Q, start)
    collect_dataset = CollectDataset()
    core = Core(agent, mdp, callbacks_fit=[collect_dataset, collect_max_Q])

    # Train
    core.learn(n_steps=10000, n_steps_per_fit=1, quiet=True)

    return collect_dataset.get().reward, collect_max_Q.get()


if __name__ == '__main__':
    n_experiment = 10000
    algorithms = [QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning, SARSA]
    exponents = [1, .8]
    window = 100

    logger = Logger('van_hasselt_double_q', results_dir=get_log_dir(__file__))
    logger.log_experiment_info(QLearning, n_experiment=n_experiment, exponents=exponents,
                               smoothing_window=window)

    for exp in exponents:
        curves = dict()

        for algorithm_class in algorithms:
            logger.info(f'Algorithm: {algorithm_class.name()}, decay exponent: {exp}')

            out = Parallel(n_jobs=-1)(delayed(experiment)(algorithm_class, exp, seed)
                                      for seed in range(n_experiment))
            r = np.array([o[0] for o in out])
            max_Qs = np.array([o[1] for o in out])

            r = np.convolve(r.mean(0), np.ones(window) / window, 'valid')
            max_Qs = max_Qs.mean(0)

            name = f'{algorithm_class.name()}_{exp}'
            logger.log_numpy_array(**{name + '_r': r, name + '_maxQ': max_Qs})

            curves[algorithm_class.name()] = (r, max_Qs)

        fig = plot_results(curves, exp, window)
        fig.savefig(logger.path / f'grid_world_{exp}.png')
