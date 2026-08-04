"""
Simple script to solve a double chain with Q-Learning and some of its variants.
The considered double chain is the one presented in:
"Relative Entropy Policy Search". Peters J. et al. 2010.

"""
import numpy as np
from joblib import Parallel, delayed

from mushroom_rl.algorithms.value import QLearning, DoubleQLearning, \
    WeightedQLearning, SpeedyQLearning
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import FiniteMDP
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.callbacks import CollectQ
from mushroom_rl.rl_utils.parameters import Parameter, DecayParameter
from mushroom_rl.utils.experiments import get_data_dir, get_log_dir


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
        for algorithm_class in algorithms:
            logger.info(f'Algorithm: {algorithm_class.name()}, decay exponent: {exp}')

            out = Parallel(n_jobs=1)(delayed(experiment)(algorithm_class, exp, seed)
                                     for seed in range(n_experiment))
            Qs = np.array(out).mean(0)

            logger.log_numpy_array(**{f'{algorithm_class.name()}_{exp}': Qs[:, 0, 0]})
