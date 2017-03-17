import inspect

from PyPi import algorithms as algs
from PyPi import approximators as apprxs
from PyPi import environments as envs
from PyPi import policy as pi


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
