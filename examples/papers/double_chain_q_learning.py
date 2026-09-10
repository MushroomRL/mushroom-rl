"""
Simple script to solve a double chain with Q-Learning and some of its variants.
The considered double chain is the one presented in:
"Relative Entropy Policy Search". Peters J. et al. 2010.

"""
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

from mushroom_rl.algorithms.value import QLearning, DoubleQLearning, \
    WeightedQLearning, SpeedyQLearning
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import FiniteMDP
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.callbacks import CollectQ
from mushroom_rl.rl_utils.parameters import Parameter, DecayParameter
from mushroom_rl.utils.experiments import get_data_dir, get_log_dir

matplotlib.use('Agg')


def plot_results(curves, exp):
    """
    Draw the action value of the first action in the initial state against the learning steps, one line per
    algorithm.

    Args:
        curves (dict): mapping from algorithm name to the curve of the action value;
        exp (float): the decay exponent of the learning rate the curves were collected with.

    Returns:
        The figure holding the plot.

    """
    fig, ax = plt.subplots()
    fig.suptitle(f'Learning rate decaying as 1 / n^{exp}')

    for name, q in curves.items():
        ax.plot(np.arange(1, len(q) + 1), q, label=name)

    ax.set_xlabel('steps')
    ax.set_ylabel('Q(s0, a0)')
    ax.legend()

    return fig


def experiment(algorithm_class, exp, seed):
    np.random.seed(seed)

    # MDP
    path = get_data_dir(__file__) / 'double_chain'
    p = np.load(path / 'p.npy')
    rew = np.load(path / 'rew.npy')
    mdp = FiniteMDP(p, rew, gamma=.9)

    # Policy
    epsilon = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = DecayParameter(value=1., exp=exp, shape=mdp.info.size)
    agent = algorithm_class(mdp.info, pi, learning_rate=learning_rate)

    # Algorithm
    collect_Q = CollectQ(agent.Q)
    core = Core(agent, mdp, callbacks_fit=[collect_Q])

    # Train
    core.learn(n_steps=20000, n_steps_per_fit=1, quiet=True)

    return collect_Q.get()


if __name__ == '__main__':
    n_experiment = 5
    algorithms = [QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning]
    exponents = [1, .51]

    logger = Logger('double_chain_q_learning', results_dir=get_log_dir(__file__))
    logger.log_experiment_info(QLearning, n_experiment=n_experiment, exponents=exponents)

    for exp in exponents:
        curves = dict()

        for algorithm_class in algorithms:
            logger.info(f'Algorithm: {algorithm_class.name()}, decay exponent: {exp}')

            out = Parallel(n_jobs=-1)(delayed(experiment)(algorithm_class, exp, seed)
                                      for seed in range(n_experiment))
            Qs = np.array(out).mean(0)

            logger.log_numpy_array(**{f'{algorithm_class.name()}_{exp}': Qs[:, 0, 0]})

            curves[algorithm_class.name()] = Qs[:, 0, 0]

        fig = plot_results(curves, exp)
        fig.savefig(logger.path / f'double_chain_{exp}.png')
