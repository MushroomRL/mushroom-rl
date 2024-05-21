import argparse
import datetime
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.value import AveragedDQN, CategoricalDQN, DQN,\
    DoubleDQN, MaxminDQN, DuelingDQN, NoisyDQN, Rainbow
from mushroom_rl.approximators.parametric import NumpyTorchApproximator
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.habitat_env import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import LinearParameter, Parameter
from mushroom_rl.rl_utils.replay_memory import PrioritizedReplayMemory


"""
This script runs the Habitat navigation task in a Replica scene with DQN.
By default, the scene is apartment_0 and start / goal location are defined in
`replica_train_apartment-0.json`.

"""

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Network(nn.Module):
    n_features = 512

    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[0]
        n_output = output_shape[0]

        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(n_input, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
        )

        dummy_obs = torch.zeros(1, *input_shape)
        conv_out_size = np.prod(self.feat_extract(dummy_obs).shape)

        self.fully_connect = nn.Sequential(
            init_(nn.Linear(conv_out_size, self.n_features)),
            nn.ReLU(),
            init_(nn.Linear(self.n_features, self.n_features)),
            nn.ReLU()
        )

        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.output_layer = init_(nn.Linear(self.n_features, n_output))

    def forward(self, state, action=None):
        q = self.feat_extract(state.float() / 255.)
        q = self.fully_connect(q.view(state.shape[0], -1))
        q = self.output_layer(q)

        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))
            return q_acted



class FeatureNetwork(nn.Module):
    n_features = 512

    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[0]

        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(n_input, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
        )

        dummy_obs = torch.zeros(1, *input_shape)
        conv_out_size = np.prod(self.feat_extract(dummy_obs).shape)

        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.output_layer = init_(nn.Linear(conv_out_size, self.n_features))

    def forward(self, state, action=None):
        h = self.feat_extract(state.float() / 255.)
        h = self.output_layer(h.view(state.shape[0], -1))

        return h



def print_epoch(epoch, logger):
    logger.info('################################################################')
    logger.info('Epoch: %d' % epoch)
    logger.info('----------------------------------------------------------------')


def get_stats(dataset, logger):
    score = dataset.compute_metrics()
    logger.info(('min_reward: %f, max_reward: %f, mean_reward: %f,'
                ' median_reward: %f, episodes_completed: %d' % score))

    return score


def experiment():
    np.random.seed()

    # Argument parser
    parser = argparse.ArgumentParser()

    arg_mem = parser.add_argument_group('Replay Memory')
    arg_mem.add_argument("--initial-replay-size", type=int, default=50000,
                         help='Initial size of the replay memory.')
    arg_mem.add_argument("--max-replay-size", type=int, default=500000,
                         help='Max size of the replay memory.')
    arg_mem.add_argument("--prioritized", action='store_true',
                         help='Whether to use prioritized memory or not.')

    arg_net = parser.add_argument_group('Deep Q-Network')
    arg_net.add_argument("--optimizer",
                         choices=['adadelta',
                                  'adam',
                                  'rmsprop',
                                  'rmspropcentered'],
                         default='adam',
                         help='Name of the optimizer to use.')
    arg_net.add_argument("--learning-rate", type=float, default=.0001,
                         help='Learning rate value of the optimizer.')
    arg_net.add_argument("--decay", type=float, default=.95,
                         help='Discount factor for the history coming from the'
                              'gradient momentum in rmspropcentered and'
                              'rmsprop')
    arg_net.add_argument("--epsilon", type=float, default=1e-8,
                         help='Epsilon term used in rmspropcentered and'
                              'rmsprop')

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--algorithm", choices=['dqn', 'ddqn', 'adqn', 'mmdqn',
                                                 'cdqn', 'dueldqn', 'ndqn', 'rainbow'],
                         default='dqn',
                         help='Name of the algorithm. dqn is for standard'
                              'DQN, ddqn is for Double DQN and adqn is for'
                              'Averaged DQN.')
    arg_alg.add_argument("--n-approximators", type=int, default=1,
                         help="Number of approximators used in the ensemble for"
                              "AveragedDQN or MaxminDQN.")
    arg_alg.add_argument("--batch-size", type=int, default=32,
                         help='Batch size for each fit of the network.')
    arg_alg.add_argument("--history-length", type=int, default=4,
                         help='Number of frames composing a state.')
    arg_alg.add_argument("--target-update-frequency", type=int, default=10000,
                         help='Number of collected samples before each update'
                              'of the target network.')
    arg_alg.add_argument("--evaluation-frequency", type=int, default=250000,
                         help='Number of collected samples before each'
                              'evaluation. An epoch ends after this number of'
                              'steps')
    arg_alg.add_argument("--train-frequency", type=int, default=4,
                         help='Number of collected samples before each fit of'
                              'the neural network.')
    arg_alg.add_argument("--max-steps", type=int, default=5000000,
                         help='Total number of collected samples.')
    arg_alg.add_argument("--final-exploration-frame", type=int, default=10000000,
                         help='Number of collected samples until the exploration'
                              'rate stops decreasing.')
    arg_alg.add_argument("--initial-exploration-rate", type=float, default=1.,
                         help='Initial value of the exploration rate.')
    arg_alg.add_argument("--final-exploration-rate", type=float, default=.1,
                         help='Final value of the exploration rate. When it'
                              'reaches this values, it stays constant.')
    arg_alg.add_argument("--test-exploration-rate", type=float, default=.05,
                         help='Exploration rate used during evaluation.')
    arg_alg.add_argument("--test-episodes", type=int, default=5,
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
                         help='Sigma0 coefficient for noise initialization in'
                              'NoisyDQN and Rainbow.')

    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument('--use-cuda', action='store_true',
                           help='Flag specifying whether to use the GPU.')
    arg_utils.add_argument('--save', action='store_true',
                           help='Flag specifying whether to save the model.')
    arg_utils.add_argument('--load-path', type=str,
                           help='Path of the model to be loaded.')
    arg_utils.add_argument('--render', action='store_true',
                           help='Flag specifying whether to render the grid.')
    arg_utils.add_argument('--quiet', action='store_true',
                           help='Flag specifying whether to hide the progress'
                                'bar.')
    arg_utils.add_argument('--debug', action='store_true',
                           help='Flag specifying whether the script has to be'
                                'run in debug mode.')

    args = parser.parse_args()

    scores = list()

    optimizer = dict()
    if args.optimizer == 'adam':
        optimizer['class'] = optim.Adam
        optimizer['params'] = dict(lr=args.learning_rate,
                                   eps=args.epsilon)
    elif args.optimizer == 'adadelta':
        optimizer['class'] = optim.Adadelta
        optimizer['params'] = dict(lr=args.learning_rate,
                                   eps=args.epsilon)
    elif args.optimizer == 'rmsprop':
        optimizer['class'] = optim.RMSprop
        optimizer['params'] = dict(lr=args.learning_rate,
                                   alpha=args.decay,
                                   eps=args.epsilon)
    elif args.optimizer == 'rmspropcentered':
        optimizer['class'] = optim.RMSprop
        optimizer['params'] = dict(lr=args.learning_rate,
                                   alpha=args.decay,
                                   eps=args.epsilon,
                                   centered=True)
    else:
        raise ValueError

    # Summary folder
    folder_name = './logs/habitat_nav_' + args.algorithm +\
        '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    pathlib.Path(folder_name).mkdir(parents=True)

    # Settings
    if args.debug:
        initial_replay_size = 50
        max_replay_size = 500
        train_frequency = 5
        target_update_frequency = 10
        test_episodes = 20
        evaluation_frequency = 50
        max_steps = 1000
    else:
        initial_replay_size = args.initial_replay_size
        max_replay_size = args.max_replay_size
        train_frequency = args.train_frequency
        target_update_frequency = args.target_update_frequency
        test_episodes = args.test_episodes
        evaluation_frequency = args.evaluation_frequency
        max_steps = args.max_steps

    # MDP
    config_file = os.path.join(pathlib.Path(__file__).parent.resolve(),
        'pointnav_apartment-0.yaml') # Custom task for Replica scenes
    wrapper = 'HabitatNavigationWrapper'
    mdp = Habitat(wrapper, config_file)
    opt_return = mdp.env.get_optimal_policy_return()

    if args.load_path:
        logger = Logger(DQN.__name__, results_dir=None)
        logger.strong_line()
        logger.info('Optimal Policy Undiscounted Return: ' + str(opt_return))
        logger.info('Experiment Algorithm: ' + DQN.__name__)

        # Agent
        agent = DQN.load(args.load_path)
        epsilon_test = Parameter(value=args.test_exploration_rate)
        agent.policy.set_epsilon(epsilon_test)

        # Algorithm
        core_test = Core(agent, mdp)

        # Evaluate model
        dataset = core_test.evaluate(n_episodes=args.test_episodes,
                                     render=args.render,
                                     quiet=args.quiet)
        get_stats(dataset, logger)
    else:
        # Policy
        epsilon = LinearParameter(value=args.initial_exploration_rate,
                                  threshold_value=args.final_exploration_rate,
                                  n=args.final_exploration_frame)
        epsilon_test = Parameter(value=args.test_exploration_rate)
        epsilon_random = Parameter(value=1)
        pi = EpsGreedy(epsilon=epsilon_random)

        # Approximator
        approximator_params = dict(
            network=Network if args.algorithm not in ['dueldqn', 'cdqn', 'ndqn', 'rainbow'] else FeatureNetwork,
            input_shape=mdp.info.observation_space.shape,
            output_shape=(mdp.info.action_space.n,),
            n_actions=mdp.info.action_space.n,
            n_features=Network.n_features,
            optimizer=optimizer
        )
        if args.algorithm not in ['cdqn', 'rainbow']:
            approximator_params['loss'] = F.smooth_l1_loss

        approximator = NumpyTorchApproximator

        if args.prioritized:
            replay_memory = PrioritizedReplayMemory(
                initial_replay_size, max_replay_size, alpha=args.alpha_coeff,
                beta=LinearParameter(.4, threshold_value=1,
                                     n=max_steps // train_frequency)
            )
        else:
            replay_memory = None

        # Agent
        algorithm_params = dict(
            batch_size=args.batch_size,
            target_update_frequency=target_update_frequency // train_frequency,
            replay_memory=replay_memory,
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size
        )

        if args.algorithm == 'dqn':
            alg = DQN
            agent = alg(mdp.info, pi, approximator,
                        approximator_params=approximator_params,
                        **algorithm_params)
        elif args.algorithm == 'ddqn':
            alg = DoubleDQN
            agent = alg(mdp.info, pi, approximator,
                        approximator_params=approximator_params,
                        **algorithm_params)
        elif args.algorithm == 'adqn':
            alg = AveragedDQN
            agent = alg(mdp.info, pi, approximator,
                        approximator_params=approximator_params,
                        n_approximators=args.n_approximators,
                        **algorithm_params)
        elif args.algorithm == 'mmdqn':
            alg = MaxminDQN
            agent = alg(mdp.info, pi, approximator,
                        approximator_params=approximator_params,
                        n_approximators=args.n_approximators,
                        **algorithm_params)
        elif args.algorithm == 'dueldqn':
            alg = DuelingDQN
            agent = alg(mdp.info, pi, approximator_params=approximator_params,
                        **algorithm_params)
        elif args.algorithm == 'cdqn':
            alg = CategoricalDQN
            agent = alg(mdp.info, pi, approximator_params=approximator_params,
                        n_atoms=args.n_atoms, v_min=args.v_min,
                        v_max=args.v_max, **algorithm_params)
        elif args.algorithm == 'ndqn':
            alg = NoisyDQN
            agent = alg(mdp.info, pi, approximator_params=approximator_params,
                        sigma_coeff=args.sigma_coeff, **algorithm_params)
        elif args.algorithm == 'rainbow':
            alg = Rainbow
            beta = LinearParameter(.4, threshold_value=1, n=max_steps // train_frequency)
            agent = alg(mdp.info, pi, approximator_params=approximator_params,
                        n_atoms=args.n_atoms, v_min=args.v_min,
                        v_max=args.v_max, n_steps_return=args.n_steps_return,
                        alpha_coeff=args.alpha_coeff, beta=beta,
                        sigma_coeff=args.sigma_coeff, **algorithm_params)

        logger = Logger(alg.__name__, results_dir=None)
        logger.strong_line()
        logger.info('Optimal Policy Undiscounted Return: ' + str(opt_return))
        logger.info('Experiment Algorithm: ' + alg.__name__)

        # Algorithm
        core = Core(agent, mdp)

        # RUN

        # Fill replay memory with random dataset
        print_epoch(0, logger)
        core.learn(n_steps=initial_replay_size,
                   n_steps_per_fit=initial_replay_size, quiet=args.quiet)

        if args.save:
            agent.save(folder_name + '/agent_0.msh')

        # Evaluate initial policy
        pi.set_epsilon(epsilon_test)
        dataset = core.evaluate(n_episodes=test_episodes, render=args.render,
                                quiet=args.quiet)
        scores.append(get_stats(dataset, logger))

        np.save(folder_name + '/scores.npy', scores)
        for n_epoch in range(1, max_steps // evaluation_frequency + 1):
            print_epoch(n_epoch, logger)
            logger.info('- Learning:')
            # learning step
            pi.set_epsilon(epsilon)
            core.learn(n_steps=evaluation_frequency,
                       n_steps_per_fit=train_frequency, quiet=args.quiet)

            if args.save:
                agent.save(folder_name + '/agent_' + str(n_epoch) + '.msh')

            logger.info('- Evaluation:')
            # evaluation step
            pi.set_epsilon(epsilon_test)
            dataset = core.evaluate(n_episodes=test_episodes, render=args.render,
                                    quiet=args.quiet)
            scores.append(get_stats(dataset, logger))

            np.save(folder_name + '/scores.npy', scores)

    return scores


if __name__ == '__main__':
    experiment()
