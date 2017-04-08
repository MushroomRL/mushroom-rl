import argparse
import inspect
import json

from PyPi import algorithms as algs
from PyPi import approximators as apprxs
from PyPi import environments as envs
from PyPi import policy as pi
from PyPi.utils import logger
from PyPi.utils import spaces


def load_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='The path of the experiment'
                                                   'configuration file.')
    parser.add_argument('--logging', default=1, type=int, help='Logging level.')
    args = parser.parse_args()

    # Load config file
    if args.config is not None:
        load_path = args.config
        with open(load_path) as f:
            config = json.load(f)
    else:
        raise ValueError('Configuration file path missing.')

    # Logger
    logger.Logger(args.logging)

    # MDP
    mdp = get_environment(config['environment']['name'],
                          **config['environment']['params'])

    # Policy
    policy = get_policy(config['policy']['name'],
                        **config['policy']['params'])

    # Regressor
    if 'approximator' in config:
        approximator = get_approximator(config['approximator']['name'],
                                        **config['approximator']['params'])
        if config['approximator']['action_regression']:
            if isinstance(mdp.action_space, spaces.Discrete) or \
                    isinstance(mdp.action_space, spaces.DiscreteValued) or \
                    isinstance(mdp.action_space, spaces.MultiDiscrete):
                approximator = apprxs.ActionRegressor(approximator,
                                                      mdp.action_space.values)
            else:
                raise ValueError('Action regression cannot be done with continuous'
                                 ' action spaces.')
    else:
        approximator = None

    return mdp, policy, approximator, config


def load_class(module, name, instantiate=True, **params):
    members = inspect.getmembers(module, inspect.isclass)
    for n, obj in members:
        if name == n:
            if instantiate:
                return obj(**params)
            else:
                return obj
    raise ValueError(name + ' class not exists.')


def get_algorithm(name, agent, mdp, **algorithm_params):
    algorithm_params['agent'] = agent
    algorithm_params['mdp'] = mdp
    return load_class(algs, name, **algorithm_params)


def get_approximator(name, **approximator_params):
    return apprxs.Regressor(approximator_class=load_class(apprxs, name, False),
                            **approximator_params)


def get_environment(name, **environment_params):
    return load_class(envs, name, **environment_params)


def get_policy(name, **policy_params):
    return load_class(pi, name, **policy_params)
