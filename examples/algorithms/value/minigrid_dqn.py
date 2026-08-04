"""
This script runs MiniGrid tasks with DQN and some of its variants.

"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mushroom_rl.algorithms.value import AveragedDQN, CategoricalDQN, DQN, \
    DoubleDQN, MaxminDQN, DuelingDQN, NoisyDQN, Rainbow
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import MiniGridRGB
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import LinearParameter, Parameter
from mushroom_rl.rl_utils.replay_memory import PrioritizedReplayMemory
from mushroom_rl.utils.torch_utils import TorchUtils
from mushroom_rl.utils.experiments import get_log_dir, select_class


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features=512, obs_high=255., **kwargs):
        super().__init__()

        n_output = output_shape[0]

        self._phi = FeatureNetwork(input_shape, (n_features,), obs_high=obs_high)
        self._h5 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h5.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None):
        h = self._phi(state)
        q = self._h5(h)

        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))

            return q_acted


class FeatureNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, obs_high=255., **kwargs):
        super().__init__()

        n_input = input_shape[0]
        n_output = output_shape[0]

        self._obs_high = obs_high

        self._h1 = nn.Conv2d(n_input, 32, kernel_size=3, stride=2, padding=1)
        self._h2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self._h3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        conv_out_size = TorchUtils.compute_flat_output_size(nn.Sequential(self._h1, self._h2, self._h3),
                                                            input_shape)
        self._h4 = nn.Linear(conv_out_size, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, state, action=None):
        h = F.relu(self._h1(state.float() / self._obs_high))
        h = F.relu(self._h2(h))
        h = F.relu(self._h3(h))
        return F.relu(self._h4(h.view(state.shape[0], -1)))


def get_algorithms():
    """
    Return the DQN variants the script can run.

    """
    return [DQN, DoubleDQN, AveragedDQN, MaxminDQN, DuelingDQN, CategoricalDQN, NoisyDQN, Rainbow]


def get_settings(args):
    """
    Return the run length settings, shrunk to a few hundred steps in debug mode.

    """
    if args.debug:
        return dict(initial_replay_size=50, max_replay_size=500, train_frequency=5,
                    target_update_frequency=10, test_episodes=20, evaluation_frequency=50, max_steps=1000)

    return dict(initial_replay_size=args.initial_replay_size, max_replay_size=args.max_replay_size,
                train_frequency=args.train_frequency, target_update_frequency=args.target_update_frequency,
                test_episodes=args.test_episodes, evaluation_frequency=args.evaluation_frequency,
                max_steps=args.max_steps)


def build_beta(settings):
    """
    Build the importance sampling exponent of the prioritized replay memory, annealed to 1 over training.

    """
    return LinearParameter(.4, threshold_value=1,
                           n=settings['max_steps'] // settings['train_frequency'])


def build_approximator_params(alg, args, mdp):
    """
    Build the approximator parameters. The variants replacing the output layer of the network build on
    the feature extractor alone, and the ones predicting a distribution instead of an action value bring
    their own loss.

    Any variant can be run on an ensemble of approximators, which for most of them only averages
    independently initialized networks. The two variants building the ensemble themselves are excluded,
    as they set the number of models from their own parameter.

    """
    feature_algorithms = [DuelingDQN, CategoricalDQN, NoisyDQN, Rainbow]
    distributional_algorithms = [CategoricalDQN, Rainbow]
    ensemble_algorithms = [AveragedDQN, MaxminDQN]

    approximator_params = dict(
        network=FeatureNetwork if alg in feature_algorithms else Network,
        input_shape=(args.history_length,) + mdp.info.observation_space.shape,
        output_shape=(mdp.info.action_space.n,),
        n_actions=mdp.info.action_space.n,
        n_features=512,
        obs_high=float(mdp.info.observation_space.high.max()),
        optimizer=TorchUtils.get_optimizer(args.optimizer, args.learning_rate, eps=args.epsilon,
                                           decay=args.decay),
    )

    if alg not in distributional_algorithms:
        approximator_params['loss'] = F.smooth_l1_loss

    if args.n_approximators and alg not in ensemble_algorithms:
        approximator_params['n_models'] = args.n_approximators

    return approximator_params


def build_algorithm_params(args, settings):
    """
    Build the parameters shared by every DQN variant, including the replay memory.

    """
    if args.prioritized:
        replay_memory = {"class": PrioritizedReplayMemory,
                         "params": {"alpha": args.alpha_coeff, "beta": build_beta(settings)}}
    else:
        replay_memory = None

    return dict(
        batch_size=args.batch_size,
        target_update_frequency=settings['target_update_frequency'] // settings['train_frequency'],
        replay_memory=replay_memory,
        initial_replay_size=settings['initial_replay_size'],
        max_replay_size=settings['max_replay_size'],
        history_length=args.history_length,
    )


def build_variant_params(alg, args, settings):
    """
    Build the parameters specific to the selected DQN variant, on top of the shared ones.

    """
    if alg in (AveragedDQN, MaxminDQN):
        return dict(n_approximators=args.n_approximators) if args.n_approximators else dict()
    elif alg is CategoricalDQN:
        return dict(n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max)
    elif alg is NoisyDQN:
        return dict(sigma_coeff=args.sigma_coeff)
    elif alg is Rainbow:
        return dict(n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max,
                    n_steps_return=args.n_steps_return, alpha_coeff=args.alpha_coeff,
                    beta=build_beta(settings), sigma_coeff=args.sigma_coeff)

    return dict()


def build_agent(alg, args, settings, mdp, pi):
    """
    Build the agent of the selected DQN variant.

    """
    approximator_params = build_approximator_params(alg, args, mdp)
    params = build_algorithm_params(args, settings) | build_variant_params(alg, args, settings)

    return alg(mdp.info, pi, approximator_params=approximator_params, **params)


def evaluate(core, logger, epoch, test_episodes, args):
    dataset = core.evaluate(n_episodes=test_episodes, render=args.render, quiet=args.quiet)
    score = dataset.compute_metrics()

    logger.log_evaluation(epoch, min_reward=score[0], max_reward=score[1], mean_reward=score[2],
                          median_reward=score[3], episodes_completed=score[4])

    return score


def experiment(args, seed=None):
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    settings = get_settings(args)

    # MDP
    mdp = MiniGridRGB(args.name)
    mdp.seed(seed)

    if args.load_path:
        logger = Logger(DQN.name(), results_dir=None)
        logger.log_experiment_info(DQN, mdp, load_path=args.load_path)

        # Agent
        agent = DQN.load(args.load_path)
        agent.policy.set_epsilon(Parameter(value=args.test_exploration_rate))

        # Algorithm
        core = Core(agent, mdp, logger=logger)

        # Evaluate model
        evaluate(core, logger, 0, args.test_episodes, args)

        return

    alg = select_class(args.algorithm, get_algorithms())

    TorchUtils.set_default_device('cuda:0' if torch.cuda.is_available() and args.use_cuda else 'cpu')

    # Policy
    epsilon = LinearParameter(value=args.initial_exploration_rate,
                              threshold_value=args.final_exploration_rate,
                              n=args.final_exploration_frame)
    epsilon_test = Parameter(value=args.test_exploration_rate)
    pi = EpsGreedy(epsilon=Parameter(value=1), backend='torch')

    # Agent
    agent = build_agent(alg, args, settings, mdp, pi)

    logger = Logger('minigrid_' + args.algorithm + '_' + args.name,
                    results_dir=get_log_dir(__file__), use_timestamp=True)
    logger.log_experiment_info(alg, mdp, batch_size=args.batch_size, history_length=args.history_length,
                               optimizer=args.optimizer, learning_rate=args.learning_rate,
                               prioritized=args.prioritized, **settings)

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    # Fill replay memory with random dataset
    core.learn(n_steps=settings['initial_replay_size'],
               n_steps_per_fit=settings['initial_replay_size'], quiet=args.quiet)

    if args.save:
        logger.log_agent(agent, epoch=0)

    # Evaluate initial policy
    pi.set_epsilon(epsilon_test)
    evaluate(core, logger, 0, settings['test_episodes'], args)

    for n_epoch in range(1, settings['max_steps'] // settings['evaluation_frequency'] + 1):
        # learning step
        pi.set_epsilon(epsilon)
        core.learn(n_steps=settings['evaluation_frequency'],
                   n_steps_per_fit=settings['train_frequency'], quiet=args.quiet)

        if args.save:
            logger.log_agent(agent, epoch=n_epoch)

        # evaluation step
        pi.set_epsilon(epsilon_test)
        evaluate(core, logger, n_epoch, settings['test_episodes'], args)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    arg_env = parser.add_argument_group('Environment')
    arg_env.add_argument("--name", type=str, default='MiniGrid-Empty-5x5-v0',
                         help='Gymnasium ID of the MiniGrid environment.')

    arg_mem = parser.add_argument_group('Replay Memory')
    arg_mem.add_argument("--initial-replay-size", type=int, default=5_000, help='Initial size of the replay memory.')
    arg_mem.add_argument("--max-replay-size", type=int, default=500_000, help='Max size of the replay memory.')
    arg_mem.add_argument("--prioritized", action='store_true', help='Whether to use prioritized memory or not.')

    arg_net = parser.add_argument_group('Deep Q-Network')
    arg_net.add_argument("--optimizer", choices=['adadelta', 'adam', 'rmsprop', 'rmspropcentered'],
                         default='adam', help='Name of the optimizer to use.')
    arg_net.add_argument("--learning-rate", type=float, default=.0001, help='Learning rate value of the optimizer.')
    arg_net.add_argument("--decay", type=float, default=.95,
                         help='Discount factor for the history coming from the gradient momentum in '
                              'rmspropcentered and rmsprop')
    arg_net.add_argument("--epsilon", type=float, default=1e-8, help='Epsilon term used in rmspropcentered and rmsprop')

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--algorithm", choices=[alg.name() for alg in get_algorithms()], default=DQN.name(),
                         help='Name of the algorithm to run.')
    arg_alg.add_argument("--n-approximators", type=int,
                         help="Number of approximators used in the ensemble. AveragedDQN and MaxminDQN need "
                              "it for their update rule; the other variants only average independently "
                              "initialized networks. Defaults to the value of the selected algorithm.")
    arg_alg.add_argument("--batch-size", type=int, default=32, help='Batch size for each fit of the network.')
    arg_alg.add_argument("--history-length", type=int, default=4, help='Number of frames composing a state.')
    arg_alg.add_argument("--target-update-frequency", type=int, default=10_000,
                         help='Number of collected samples before each update of the target network.')
    arg_alg.add_argument("--evaluation-frequency", type=int, default=250_000,
                         help='Number of collected samples before each evaluation. An epoch ends after '
                              'this number of steps')
    arg_alg.add_argument("--train-frequency", type=int, default=4,
                         help='Number of collected samples before each fit of the neural network.')
    arg_alg.add_argument("--max-steps", type=int, default=5_000_000, help='Total number of collected samples.')
    arg_alg.add_argument("--final-exploration-frame", type=int, default=1_000_000,
                         help='Number of collected samples until the exploration rate stops decreasing.')
    arg_alg.add_argument("--initial-exploration-rate", type=float, default=1.,
                         help='Initial value of the exploration rate.')
    arg_alg.add_argument("--final-exploration-rate", type=float, default=.1,
                         help='Final value of the exploration rate. When it reaches this values, it stays '
                              'constant.')
    arg_alg.add_argument("--test-exploration-rate", type=float, default=.05,
                         help='Exploration rate used during evaluation.')
    arg_alg.add_argument("--test-episodes", type=int, default=10,
                         help='Number of episodes for each evaluation.')
    arg_alg.add_argument("--alpha-coeff", type=float, default=.6,
                         help='Prioritization exponent for prioritized experience replay.')
    arg_alg.add_argument("--n-atoms", type=int, default=51,
                         help='Number of atoms for Categorical DQN.')
    arg_alg.add_argument("--v-min", type=int, default=-10,
                         help='Minimum action-value for Categorical DQN.')
    arg_alg.add_argument("--v-max", type=int, default=10,
                         help='Maximum action-value for Categorical DQN.')
    arg_alg.add_argument("--n-steps-return", type=int, default=3,
                         help='Number of steps for n-step return for Rainbow.')
    arg_alg.add_argument("--sigma-coeff", type=float, default=.5,
                         help='Sigma0 coefficient for noise initialization in NoisyDQN and Rainbow.')

    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument('--use-cuda', action='store_true',
                           help='Flag specifying whether to use the GPU.')
    arg_utils.add_argument('--save', action='store_true',
                           help='Flag specifying whether to save the model.')
    arg_utils.add_argument('--load-path', type=str, help='Path of the model to be loaded.')
    arg_utils.add_argument('--render', action='store_true',
                           help='Flag specifying whether to render the grid.')
    arg_utils.add_argument('--quiet', action='store_true',
                           help='Flag specifying whether to hide the progress bar.')
    arg_utils.add_argument('--debug', action='store_true',
                           help='Flag specifying whether the script has to be run in debug mode.')

    arg_utils.add_argument('--seed', type=int, default=None,
                           help='Seed of the experiment. Random when not given.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    experiment(args, seed=args.seed)
