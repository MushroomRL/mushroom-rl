"""
This script trains the Unitree Go2 to walk with RudinPPO: the full reward set, the biased velocity command
sampling, an asymmetric actor-critic, an eight-step observation history, and a curriculum that tightens the
task as training goes on.

The curriculum deliberately lives here rather than in the environment. `Go2Isaac` only accepts new values --
for the velocity command ranges, the tolerance of the tracking rewards and the ceiling on the actuation
delay -- and the decision of when to change them is the training script's.

The asymmetry is the other half of the training setup: the robot can measure neither its own linear velocity,
nor the height it holds its trunk at, nor how late its actions arrive and how far its joint encoders are out
of calibration, so the policy is given none of them, while the critic, which only ever runs in simulation, is
given all four.

"""
import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange

from mushroom_rl.core import Core, Logger
from mushroom_rl.algorithms.actor_critic import RudinPPO
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils.isaac_sim import IsaacLauncher
from mushroom_rl.utils import TorchUtils
from mushroom_rl.utils.experiments import get_log_dir
from mushroom_rl.approximators.parametric.networks import ActorNetwork

# Isaac Sim has to be running before its environments can be imported
IsaacLauncher.launch(headless=True)

# mute an INFO log that garbles the tqdm bars
logging.getLogger("isaacsim.asset.transformer.rules.utils").setLevel(logging.WARNING)

from mushroom_rl.environments.isaacsim_envs import Go2Isaac


class PolicyNetwork(ActorNetwork):
    """
    An actor network reading only part of the observation, so that the policy is trained on what the real
    robot can measure while the critic keeps the entries only the simulation knows.

    """
    def __init__(self, input_shape, output_shape, observed_indices, **kwargs):
        """
        Constructor.

        Args:
            input_shape (tuple): shape of the network input, whose last axis is the full observation and whose
                leading axes, if any, are the stacked history;
            output_shape (tuple): shape of the output (e.g. the action);
            observed_indices (torch.tensor): the entries of the observation the policy is allowed to see;
            **kwargs: other parameters of :class:`ActorNetwork`.

        """
        super().__init__(tuple(input_shape[:-1]) + (len(observed_indices),), output_shape, **kwargs)

        self.register_buffer('_observed_indices', observed_indices)

    def forward(self, state, **kwargs):
        return super().forward(state[..., self._observed_indices], **kwargs)


class Curriculum:
    """
    The schedule the task is tightened along, applied after every policy update.

    Two schedules run on their own thresholds, so that the shift in the reward function and the widening of
    the commands do not reach the critic in the same window: the command ranges widen and the actuation delay
    grows in stages, while the tracking tolerance is interpolated continuously.

    """
    def __init__(self, mdp, command_steps, command_ranges, delay_steps, tracking_steps, tracking_stds):
        """
        Constructor.

        Args:
            mdp (Go2Isaac): the environment the schedule is applied to;
            command_steps (list): the step counts at which the next stage begins, one fewer than the stages;
            command_ranges (list): the linear velocity command range of each stage, applied to x and y;
            delay_steps (list): the ceiling on the actuation delay of each stage, in physics steps;
            tracking_steps (tuple): the step counts the tracking tolerance is interpolated between;
            tracking_stds (tuple): the tolerances at the two ends of that interpolation.

        """
        self._mdp = mdp
        self._command_steps = command_steps
        self._command_ranges = command_ranges
        self._delay_steps = delay_steps
        self._tracking_steps = tracking_steps
        self._tracking_stds = tracking_stds

        self._step = 0
        self.apply()

    def __call__(self, dataset):
        # a vectorized dataset holds one row per control step, whichever environment it came from
        self._step += len(dataset)
        self.apply()

    def apply(self):
        low, high = self._command_ranges[self.stage]
        self._mdp.command_ranges = dict(lin_vel_x=(low, high), lin_vel_y=(low, high))
        self._mdp.max_delay_steps = self._delay_steps[self.stage]

        start, end = self._tracking_stds
        self._mdp.tracking_stds = {name: start[name] + self.progress * (end[name] - start[name])
                                   for name in start}

    @property
    def stage(self):
        return sum(self._step >= threshold for threshold in self._command_steps)

    @property
    def progress(self):
        start, end = self._tracking_steps
        return min(max((self._step - start) / (end - start), 0.), 1.)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-render', action='store_false', dest='render',
                        help='skip rendering/recording the evaluations')

    return parser.parse_args()


def observed_indices(mdp, *privileged):
    """
    Returns the entries of the observation the policy is allowed to see, that is every one that is not
    privileged.

    """
    hidden = set(mdp.observation_indices(*privileged).tolist())

    return torch.tensor([index for index in range(mdp.info.observation_space.shape[0]) if index not in hidden],
                        device=TorchUtils.get_device())


def initial_value(agent, dataset):
    """
    Returns the mean value the critic assigns to the initial states of the dataset, stacked into the history
    the critic reads: the entry of the current step, preceded by the zeros the history manager pads the
    beginning of an episode with.

    """
    states = dataset.get_init_states()
    history = states.new_zeros((len(states), agent.history_length) + tuple(states.shape[1:]))
    history[:, -1] = states

    return agent._V(history).mean().item()


def episode_metrics(dataset):
    """
    Returns the mean length of the episodes of the dataset and the fraction of them that ended in a fall
    rather than at the horizon.

    """
    n_episodes = int(dataset.n_episodes)

    return len(dataset) / n_episodes, dataset.absorbing.sum().item() / n_episodes


def tracking_errors(mdp, dataset):
    """
    Returns the mean linear and angular velocity tracking errors over the dataset, that is how far the base
    velocity of the robot stays from the commanded one.

    Args:
        mdp (Go2Isaac): the environment the dataset was collected on;
        dataset (Dataset): the dataset to measure.

    """
    commands = dataset.state[:, mdp.observation_indices('commands')]
    lin_vel = dataset.state[:, mdp.observation_indices('base_lin_vel')]
    ang_vel = dataset.state[:, mdp.observation_indices('base_ang_vel')]

    lin_error = torch.linalg.norm(commands[:, :2] - lin_vel[:, :2], dim=1).mean().item()
    ang_error = (commands[:, 2] - ang_vel[:, 2]).abs().mean().item()

    return lin_error, ang_error


def experiment(alg, n_epochs, n_steps, n_steps_per_fit, n_episodes_test, alg_params, policy_params, mdp_params,
               curriculum_params, n_envs=4096, horizon=1000, render=True, seed=None):
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    # MDP
    mdp = Go2Isaac(n_envs, horizon, True, **mdp_params)
    curriculum = Curriculum(mdp, **curriculum_params)

    logger = Logger('RudinPPO_go2', results_dir=get_log_dir(__file__), log_console=True, use_timestamp=True)
    logger.log_experiment_info(alg, mdp, n_epochs=n_epochs, n_steps=n_steps, n_steps_per_fit=n_steps_per_fit,
                               n_episodes_test=n_episodes_test, n_envs=n_envs, horizon=horizon,
                               **alg_params, **policy_params)

    # Policy
    network_input_shape = (alg_params['history_length'],) + mdp.info.observation_space.shape

    policy = GaussianTorchPolicy(PolicyNetwork,
                                 network_input_shape,
                                 mdp.info.action_space.shape,
                                 observed_indices=observed_indices(mdp, 'base_lin_vel', 'base_pos',
                                                                   'actual_delay', 'joint_calib_offset'),
                                 **policy_params)

    # Agent
    critic_params = dict(network=ActorNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 1e-3}},
                         loss=F.mse_loss,
                         n_features=[512, 256, 128],
                         gain_scale=0.5,
                         batch_size=int((4096 * 24) / 16),
                         use_cuda=True,
                         input_shape=network_input_shape,
                         output_shape=(1,))

    agent = alg(mdp.info, policy, critic_params=critic_params, **alg_params)

    # Algorithm
    core = Core(agent, mdp, callbacks_fit=[curriculum], logger=logger)

    # RUN
    dataset = core.evaluate(n_episodes=n_episodes_test, render=render, record=render)

    J = dataset.discounted_return.mean().item()
    R = dataset.undiscounted_return.mean().item()
    E = agent.policy.entropy().item()
    V = initial_value(agent, dataset)

    L, falls = episode_metrics(dataset)
    lin_error, ang_error = tracking_errors(mdp, dataset)

    logger.log_evaluation(0, J=J, R=R, entropy=E, V=V, length=L, falls=falls, lin_error=lin_error,
                          ang_error=ang_error, stage=curriculum.stage, progress=curriculum.progress)

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)

        # record the run only once every five epochs, to keep the videos to a manageable size
        record = render and ((it + 1) % 5 == 0 or it == n_epochs - 1)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=record, record=record)

        J = dataset.discounted_return.mean().item()
        R = dataset.undiscounted_return.mean().item()
        E = agent.policy.entropy().item()
        V = initial_value(agent, dataset)

        L, falls = episode_metrics(dataset)
        lin_error, ang_error = tracking_errors(mdp, dataset)

        logger.log_evaluation(it + 1, J=J, R=R, entropy=E, V=V, length=L, falls=falls, lin_error=lin_error,
                              ang_error=ang_error, stage=curriculum.stage, progress=curriculum.progress)

        del dataset


if __name__ == '__main__':
    args = parse_args()

    TorchUtils.set_default_device('cuda:0')

    reward_weights = dict(tracking_lin_vel=1.2, tracking_ang_vel=0.6, lin_vel_z=-1.6, ang_vel_xy=-0.07,
                          torques=-1e-5, joint_acc=-2.5e-7, action_rate=-0.005, joint_pos_limits=-0.1,
                          collision=-1.0, feet_air_time=0., flat_orientation=-0.2, joint_vel_limits=-0.1,
                          power_draw=-4e-6, similar_to_default=-0.085, stand_still_deviation=-1.4,
                          base_height=-50., feet_air_time_high=0.125, feet_air_time_low=0.75,
                          feet_air_time_symmetry=-0.25, feet_clearance=0.085, feet_clearance_lateral=0.1,
                          feet_slide=-0.04, feet_slide_low=-0.2, feet_z_velocity=-0.03, long_contact=-0.7,
                          stand_still_short_contact=-0.5)

    mdp_params = dict(reward_weights=reward_weights, clamp_reward=False,
                      max_command_ranges=dict(lin_vel_x=(-2.5, 2.5), lin_vel_y=(-2.5, 2.5),
                                              ang_vel_z=(-1.5, 1.5)),
                      command_ranges=dict(ang_vel_z=(-1.5, 1.5)),
                      command_resampling_time_range=(5., 10.), rel_heading_envs=0.5, rel_standing_envs=0.1,
                      frac_rotating_envs=0.15, frac_low_speed_envs=0.35, low_speed_threshold=0.5)

    curriculum_params = dict(command_steps=[24000, 48000],
                             command_ranges=[(-1., 1.), (-2., 2.), (-2.5, 2.5)],
                             delay_steps=[2, 3, 4],
                             tracking_steps=(8000, 20000),
                             tracking_stds=(dict(lin_vel=0.5, lin_vel_slope=0., ang_vel=0.5, ang_vel_slope=0.),
                                            dict(lin_vel=0.05, lin_vel_slope=0.2, ang_vel=0.1,
                                                 ang_vel_slope=0.5)))

    ppo_params = dict(actor_optimizer={'class': optim.Adam,
                                       'params': {'lr': 1e-3}},
                      n_epochs_policy=5,
                      batch_size=int((4096 * 24) / 16),
                      eps_ppo=.2,
                      lam=.95,
                      ent_coeff=0.01,
                      history_length=8)

    policy_params = dict(std_0=1., n_features=[512, 256, 128], gain_scale=0.5, use_cuda=True)

    experiment(alg=RudinPPO, n_epochs=60, n_steps=4096 * 24 * 50, n_steps_per_fit=4096 * 24,
               n_episodes_test=256, alg_params=ppo_params, policy_params=policy_params, mdp_params=mdp_params,
               curriculum_params=curriculum_params, render=args.render)
