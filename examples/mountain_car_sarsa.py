import numpy as np
from joblib import Parallel, delayed

from mushroom.algorithms.value import TrueOnlineSARSALambda
from mushroom.core.core import Core
from mushroom.environments import *
from mushroom.features import Features
from mushroom.features.tiles import Tiles
from mushroom.policy import EpsGreedy
from mushroom.utils.callbacks import CollectDataset
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import Parameter

"""
This script aims to replicate the experiments on the Mountain Car MDP as
presented in:
"True Online TD(lambda)". Seijen H. V. et al.. 2014.

"""


def experiment(alpha):
    np.random.seed()

    # MDP
    mdp = Gym(name='MountainCar-v0', horizon=np.inf, gamma=1.)

    # Policy
    epsilon = Parameter(value=0.)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = Parameter(alpha)
    tilings = Tiles.generate(10, [10, 10],
                             mdp.info.observation_space.low,
                             mdp.info.observation_space.high)
    features = Features(tilings=tilings)

    approximator_params = dict(input_shape=(features.size,),
                               output_shape=(mdp.info.action_space.n,),
                               n_actions=mdp.info.action_space.n)
    algorithm_params = {'learning_rate': learning_rate,
                        'lambda': .9}
    fit_params = dict()
    agent_params = {'approximator_params': approximator_params,
                    'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = TrueOnlineSARSALambda(pi, mdp.info, agent_params, features)

    # Algorithm
    collect_dataset = CollectDataset()
    callbacks = [collect_dataset]
    core = Core(agent, mdp, callbacks=callbacks)

    # Train
    core.learn(n_episodes=20, n_steps_per_fit=1, render=0)

    dataset = collect_dataset.get()
    return np.mean(compute_J(dataset, 1.))


if __name__ == '__main__':
    n_experiment = 1

    alpha = .1
    Js = Parallel(
        n_jobs=-1)(delayed(experiment)(alpha) for _ in range(n_experiment))

    print(np.mean(Js))
