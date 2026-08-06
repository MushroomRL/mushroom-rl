"""
This script aims to replicate the experiments on the Taxi MDP as presented in:
"An Alternative Softmax Operator for Reinforcement Learning", Asadi K. et al. 2017.

"""
import numpy as np
from joblib import Parallel, delayed

from mushroom_rl.algorithms.value import SARSA
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Taxi
from mushroom_rl.policy import Boltzmann, EpsGreedy, Mellowmax
from mushroom_rl.utils.callbacks import CollectDataset
from mushroom_rl.rl_utils.parameters import Parameter
from mushroom_rl.utils.experiments import get_log_dir


def experiment(policy_class, value, n_steps, seed):
    np.random.seed(seed)

    # MDP
    mdp = Taxi.generate()

    # Policy
    pi = policy_class(Parameter(value=value))

    # Agent
    learning_rate = Parameter(value=.15)
    agent = SARSA(mdp.info, pi, learning_rate=learning_rate)

    # Algorithm
    collect_dataset = CollectDataset()
    core = Core(agent, mdp, callbacks_fit=[collect_dataset])

    # Train
    core.learn(n_steps=n_steps, n_steps_per_fit=1, quiet=True)

    return collect_dataset.get().reward.sum() / n_steps


if __name__ == '__main__':
    n_experiment = 25
    n_steps = 300000
    policies = [EpsGreedy, Boltzmann, Mellowmax]
    ranges = {EpsGreedy: np.linspace(.05, .5, 10),
              Boltzmann: np.linspace(.5, 10, 10),
              Mellowmax: np.linspace(.5, 10, 10)}

    logger = Logger('taxi_mellowmax', results_dir=get_log_dir(__file__))
    logger.log_experiment_info(SARSA, n_experiment=n_experiment, n_steps=n_steps)

    for policy_class in policies:
        logger.info(f'Policy: {policy_class.name()}')

        Js = list()
        for value in ranges[policy_class]:
            out = np.array(Parallel(n_jobs=-1)(delayed(experiment)(policy_class, value, n_steps, seed)
                                               for seed in range(n_experiment)))
            Js.append(out.mean())

        logger.log_numpy_array(**{policy_class.name(): np.array(Js)})
