import torch
from torch import optim
from torch import nn
import numpy as np

from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.segway import Segway
from mushroom_rl.algorithms.policy_search import ePPO
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.distributions import DiagonalGaussianTorchDistribution
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.callbacks import CollectDataset

from tqdm import tqdm, trange
tqdm.monitor_interval = 0


class LinearNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h = nn.Linear(n_input, n_output)

    def forward(self, state, **kwargs):
        a = self._h(torch.squeeze(state, 1).float())

        return a


def experiment(alg, params, n_epochs, n_episodes, n_ep_per_fit, n_ep_test):
    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    # MDP
    mdp = Segway()

    # Policy
    approximator = Regressor(TorchApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape,
                             network=LinearNetwork)

    n_weights = approximator.weights_size
    mu = torch.zeros(n_weights)
    sigma = 2e-0 * torch.ones(n_weights)
    policy = DeterministicPolicy(approximator)
    dist = DiagonalGaussianTorchDistribution(mu, sigma)

    agent = alg(mdp.info, dist, policy, **params)

    # Train
    core = Core(agent, mdp)

    dataset = core.evaluate(n_episodes=n_ep_test)
    J = dataset.discounted_return.mean()
    p = dist.get_parameters().detach().numpy()
    logger.epoch_info(0, J=J, mu=p[:n_weights], sigma=p[n_weights:])

    for i in trange(n_epochs, leave=False):
        core.learn(n_episodes=n_episodes, n_episodes_per_fit=n_ep_per_fit, render=False)
        dataset = core.evaluate(n_episodes=n_ep_test)
        J = dataset.discounted_return.mean()

        p = dist.get_parameters().detach().numpy()

        logger.epoch_info(i+1, J=J, mu=p[:n_weights], sigma=p[n_weights:])

    logger.info('Press a button to visualize the segway...')
    input()
    core.evaluate(n_episodes=3, render=True)


if __name__ == '__main__':
    eppo_params = dict(optimizer={'class': optim.Adam, 'params': {'lr': 1e-2, 'weight_decay': 0.0}},
                       n_epochs_policy=50,
                       batch_size=25,
                       eps_ppo=5e-2)

    algs_params = [
        (ePPO, eppo_params),
        ]
    for alg, params in algs_params:
        experiment(alg, params, n_epochs=20, n_episodes=100, n_ep_per_fit=25, n_ep_test=25)
