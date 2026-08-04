"""
This script shows how to solve the Segway problem with ePPO, an episode-based policy search algorithm whose
search distribution is optimized with a clipped surrogate objective.

"""
import argparse

import numpy as np
import torch
from torch import nn
from torch import optim

from tqdm import trange

from mushroom_rl.algorithms.policy_search import ePPO
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.core import Core, Logger
from mushroom_rl.distributions import DiagonalGaussianTorchDistribution
from mushroom_rl.environments import Segway
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.utils.torch_utils import TorchUtils


class LinearNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h = nn.Linear(n_input, n_output)

    def forward(self, state, **kwargs):
        a = self._h(torch.squeeze(state, 1).float())

        return a


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--use-cuda', action='store_true', help='run on the GPU instead of the CPU')
    parser.add_argument('--seed', type=int, default=None, help='seed of the experiment, random when not given')

    return parser.parse_args()


def experiment(params, n_epochs, n_episodes, n_ep_per_fit, n_ep_test, use_cuda=False, seed=None):
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    if use_cuda:
        assert torch.cuda.is_available(), 'CUDA was requested, but it is not available on this machine.'

    TorchUtils.set_default_device('cuda:0' if use_cuda else 'cpu')

    # MDP
    mdp = Segway()

    logger = Logger(ePPO.name(), results_dir=None)
    logger.log_experiment_info(ePPO, mdp, n_epochs=n_epochs, n_episodes=n_episodes,
                               n_ep_per_fit=n_ep_per_fit, n_ep_test=n_ep_test, **params)

    # Policy
    approximator = TorchApproximator(input_shape=mdp.info.observation_space.shape,
                                     output_shape=mdp.info.action_space.shape,
                                     network=LinearNetwork)

    n_weights = approximator.weights_size
    mu = torch.zeros(n_weights)
    sigma = 2e-0 * torch.ones(n_weights)
    policy = DeterministicPolicy(approximator)
    dist = DiagonalGaussianTorchDistribution(mu, sigma)

    # Agent
    agent = ePPO(mdp.info, dist, policy, **params)

    # Train
    core = Core(agent, mdp, logger=logger)

    dataset = core.evaluate(n_episodes=n_ep_test)
    J = dataset.discounted_return.mean()
    p = dist.get_parameters().detach().numpy()

    logger.log_evaluation(0, J=J, mu=p[:n_weights], sigma=p[n_weights:])

    for i in trange(n_epochs, leave=False):
        core.learn(n_episodes=n_episodes, n_episodes_per_fit=n_ep_per_fit, render=False)
        dataset = core.evaluate(n_episodes=n_ep_test)
        J = dataset.discounted_return.mean()
        p = dist.get_parameters().detach().numpy()

        logger.log_evaluation(i + 1, J=J, mu=p[:n_weights], sigma=p[n_weights:])

    # Visualize the final policy
    core.evaluate(n_episodes=3, render=True)


if __name__ == '__main__':
    args = parse_args()

    eppo_params = dict(optimizer={'class': optim.Adam, 'params': {'lr': 1e-2, 'weight_decay': 0.0}},
                       n_epochs_policy=50,
                       batch_size=25,
                       eps_ppo=5e-2)

    experiment(eppo_params, n_epochs=20, n_episodes=100, n_ep_per_fit=25, n_ep_test=25,
               use_cuda=args.use_cuda, seed=args.seed)
