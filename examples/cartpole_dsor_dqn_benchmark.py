import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from mushroom_rl.algorithms.value import DSORDQN, DoubleDQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import QNetwork
from mushroom_rl.core import Core
from mushroom_rl.environments import Gymnasium
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import LinearParameter


ALGORITHM_NAMES = ('Double DQN', 'Double SOR DQN')


def make_agent(name, mdp_info, seed, relaxation_factor):
    np.random.seed(seed)
    torch.manual_seed(seed)

    epsilon = LinearParameter(1., .05, 10_000)
    policy = EpsGreedy(epsilon, backend='torch')
    approximator_params = dict(
        network=QNetwork,
        optimizer={'class': optim.Adam, 'params': {'lr': 1e-3}},
        loss=F.smooth_l1_loss,
        input_shape=mdp_info.observation_space.shape,
        output_shape=mdp_info.action_space.size,
        n_actions=mdp_info.action_space.n,
        n_features=64,
        n_layers=2
    )
    algorithm_params = dict(
        approximator_params=approximator_params,
        batch_size=64,
        initial_replay_size=500,
        max_replay_size=20_000,
        target_update_frequency=250
    )

    if name == 'Double DQN':
        return DoubleDQN(
            mdp_info, policy, TorchApproximator, **algorithm_params)
    return DSORDQN(
        mdp_info, policy, TorchApproximator,
        relaxation_factor=relaxation_factor, **algorithm_params)


def evaluate(agent, seed, n_episodes):
    mdp = Gymnasium(
        'CartPole-v1', horizon=500, gamma=.99, headless=True)
    mdp.seed(seed)
    dataset = Core(agent, mdp).evaluate(
        n_episodes=n_episodes, greedy=True, quiet=True)
    return float(np.mean(dataset.undiscounted_return))


def run(name, seed, n_steps, checkpoint, n_eval_episodes,
        relaxation_factor=1.):
    mdp = Gymnasium(
        'CartPole-v1', horizon=500, gamma=.99, headless=True)
    mdp.seed(seed)
    agent = make_agent(name, mdp.info, seed, relaxation_factor)
    core = Core(agent, mdp)

    steps = np.arange(0, n_steps + 1, checkpoint)
    if steps[-1] != n_steps:
        steps = np.append(steps, n_steps)

    rewards = [evaluate(agent, 100_000 + seed, n_eval_episodes)]
    trained_steps = 0
    for target_steps in steps[1:]:
        core.learn(
            n_steps=int(target_steps - trained_steps),
            n_steps_per_fit=4, quiet=True)
        trained_steps = int(target_steps)
        rewards.append(evaluate(
            agent, 100_000 + seed + trained_steps, n_eval_episodes))

    mdp.stop()
    return np.column_stack((steps, rewards))


def tune(candidates, seeds, n_steps, checkpoint, n_eval_episodes):
    scores = {}
    for factor in candidates:
        curves = [
            run('Double SOR DQN', seed, n_steps, checkpoint,
                n_eval_episodes, factor)
            for seed in seeds
        ]
        scores[factor] = float(np.mean([
            np.trapezoid(curve[:, 1], curve[:, 0]) / n_steps
            for curve in curves
        ]))
    return scores


def summarize(curves):
    rewards = np.stack([curve[:, 1] for curve in curves])
    ci95 = 1.96 * rewards.std(axis=0, ddof=1) / np.sqrt(len(curves))
    steps = curves[0][:, 0]
    return {
        'steps': steps.astype(int).tolist(),
        'mean_reward': rewards.mean(axis=0).tolist(),
        'ci95_reward': ci95.tolist(),
        'seed_rewards': rewards.tolist(),
        'mean_curve_auc': float(np.mean([
            np.trapezoid(curve[:, 1], curve[:, 0]) / steps[-1]
            for curve in curves
        ])),
        'final_mean_reward': float(rewards[:, -1].mean()),
        'final_ci95_reward': float(ci95[-1]),
        'best_mean_reward': float(rewards.mean(axis=0).max())
    }


def write_csv(path, results):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([
            'algorithm', 'step', 'mean_reward', 'ci95_reward',
            'seed', 'seed_reward'
        ])
        for algorithm, result in results.items():
            seed_rewards = np.asarray(result['seed_rewards'])
            for point, step in enumerate(result['steps']):
                for seed_index, reward in enumerate(seed_rewards[:, point]):
                    writer.writerow([
                        algorithm, step, result['mean_reward'][point],
                        result['ci95_reward'][point], seed_index, reward
                    ])


def plot(path, results, selected_factor):
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    colors = {'Double DQN': 'tab:blue', 'Double SOR DQN': 'tab:orange'}
    for algorithm, result in results.items():
        steps = np.asarray(result['steps'])
        mean = np.asarray(result['mean_reward'])
        ci95 = np.asarray(result['ci95_reward'])
        label = algorithm
        if algorithm == 'Double SOR DQN':
            label += f' (w={selected_factor:g})'
        axis.plot(steps, mean, color=colors[algorithm], label=label)
        axis.fill_between(
            steps, mean - ci95, mean + ci95,
            color=colors[algorithm], alpha=.18)

    axis.axhline(475, color='0.45', linestyle='--', linewidth=1,
                 label='Solved threshold (475)')
    axis.set(
        xlabel='Environment steps', ylabel='Evaluation reward',
        xlim=(0, max(results['Double DQN']['steps'])), ylim=(0, 510))
    axis.grid(alpha=.2)
    axis.legend(loc='lower right')
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path,
                        default=Path('cartpole_dsor_results'))
    parser.add_argument('--training-steps', type=int, default=20_000)
    parser.add_argument('--tuning-steps', type=int, default=10_000)
    parser.add_argument('--checkpoint', type=int, default=1_000)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--tuning-seeds', type=int, default=3)
    parser.add_argument('--test-seeds', type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    candidates = [1., 1.01, 1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 2.]
    tuning_scores = tune(
        candidates, range(args.tuning_seeds), args.tuning_steps,
        args.checkpoint, args.eval_episodes)
    selected_factor = max(tuning_scores, key=tuning_scores.get)

    results = {}
    test_seeds = range(1000, 1000 + args.test_seeds)
    for algorithm in ALGORITHM_NAMES:
        curves = [
            run(algorithm, seed, args.training_steps, args.checkpoint,
                args.eval_episodes, selected_factor)
            for seed in test_seeds
        ]
        results[algorithm] = summarize(curves)

    summary = {
        'environment': {
            'name': 'CartPole-v1', 'horizon': 500, 'gamma': .99,
            'reward': '+1 per time step'
        },
        'protocol': {
            **vars(args),
            'output_dir': str(args.output_dir),
            'tuning_seed_range': [0, args.tuning_seeds - 1],
            'held_out_seed_range': [1000, 999 + args.test_seeds],
            'evaluation': (f'greedy policy on {args.eval_episodes} seeded '
                           'episodes per point'),
            'uncertainty': '95% confidence interval across held-out seeds'
        },
        'hyperparameters': {
            'hidden_layers': [64, 64], 'optimizer': 'Adam',
            'learning_rate': .001, 'batch_size': 64,
            'initial_replay_size': 500, 'max_replay_size': 20_000,
            'target_update_frequency': 250, 'training_frequency': 4,
            'epsilon': {'start': 1., 'end': .05, 'decay_steps': 10_000}
        },
        'tuning_scores': {
            str(factor): score for factor, score in tuning_scores.items()
        },
        'selected_relaxation_factor': selected_factor,
        'results': results
    }
    with (args.output_dir / 'summary.json').open('w') as stream:
        json.dump(summary, stream, indent=2)
    write_csv(args.output_dir / 'learning_curves.csv', results)
    plot(args.output_dir / 'reward_curve.png', results, selected_factor)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
