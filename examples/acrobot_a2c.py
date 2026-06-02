import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import A2C
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gymnasium
from mushroom_rl.policy import BoltzmannTorchPolicy
from mushroom_rl.approximators.parametric.torch_approximator import *
from mushroom_rl.rl_utils.parameters import Parameter
from mushroom_rl.approximators.parametric.networks import ActorNetwork
from tqdm import trange


def experiment(n_epochs, n_steps, n_steps_per_fit, n_step_test):
    np.random.seed()

    logger = Logger(A2C.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + A2C.__name__)

    # MDP
    horizon = 1000
    gamma = 0.99
    mdp = Gymnasium('Acrobot-v1', horizon, gamma, headless=False)

    # Policy
    policy_params = dict(
        n_features=32
    )

    beta = Parameter(1e0)
    pi = BoltzmannTorchPolicy(ActorNetwork,
                              mdp.info.observation_space.shape,
                              (mdp.info.action_space.n,),
                              beta=beta,
                              **policy_params)

    # Agent
    critic_params = dict(network=ActorNetwork,
                         optimizer={'class': optim.RMSprop,
                                    'params': {'lr': 1e-3,
                                               'eps': 1e-5}},
                         loss=F.mse_loss,
                         n_features=32,
                         batch_size=64,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    alg_params = dict(actor_optimizer={'class': optim.RMSprop,
                                       'params': {'lr': 1e-3,
                                                  'eps': 3e-3}},
                      critic_params=critic_params,
                      ent_coeff=0.01
                      )

    agent = A2C(mdp.info, pi, **alg_params)

    # Algorithm
    core = Core(agent, mdp)

    core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)

    # RUN
    dataset = core.evaluate(n_steps=n_step_test, render=False)
    R = np.mean(dataset.undiscounted_return)
    logger.epoch_info(0, R=R)

    for n in trange(n_epochs):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_steps=n_step_test, render=False)
        R = np.mean(dataset.undiscounted_return)
        logger.epoch_info(n+1, R=R)

    logger.info('Press a button to visualize acrobot')
    input()
    core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':
    experiment(n_epochs=40, n_steps=1000, n_steps_per_fit=5, n_step_test=2000)
