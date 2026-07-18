import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cartpole_dsor_dqn_benchmark import run


ALGORITHMS = ('Double DQN', 'Double SOR DQN')


def smooth(values, window):
    values = np.asarray(values, dtype=float)
    radius = window // 2
    smoothed = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index - radius)
        stop = min(len(values), index + radius + 1)
        smoothed[index] = values[start:stop].mean()
    return smoothed


def cache_name(algorithm, seed, steps, checkpoint, episodes, factor):
    short_name = 'ddqn' if algorithm == 'Double DQN' else 'dsor'
    factor_text = f'{factor:.3f}'.replace('.', 'p')
    return (f'{short_name}_seed{seed}_steps{steps}_check{checkpoint}_'
            f'episodes{episodes}_w{factor_text}.npz')


def run_and_cache(task):
    (cache_dir, algorithm, seed, steps, checkpoint, episodes,
     factor) = task
    torch.set_num_threads(1)
    cache_path = Path(cache_dir) / cache_name(
        algorithm, seed, steps, checkpoint, episodes, factor)
    if cache_path.exists():
        return str(cache_path)

    curve = run(
        algorithm, seed, steps, checkpoint, episodes,
        relaxation_factor=factor)
    np.savez_compressed(cache_path, curve=curve)
    return str(cache_path)


def collect(tasks, workers):
    paths = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_and_cache, task) for task in tasks]
        for future in as_completed(futures):
            paths.append(future.result())
    return paths


def load_curve(cache_dir, algorithm, seed, steps, checkpoint,
               episodes, factor):
    path = cache_dir / cache_name(
        algorithm, seed, steps, checkpoint, episodes, factor)
    return np.load(path)['curve']


def candidate_summary(curves, smoothing_window):
    rewards = np.stack([curve[:, 1] for curve in curves])
    smoothed = np.stack([
        smooth(seed_rewards, smoothing_window)
        for seed_rewards in rewards
    ])
    steps = curves[0][:, 0]
    seed_auc = np.asarray([
        np.trapezoid(seed_curve, steps) / steps[-1]
        for seed_curve in smoothed
    ])
    return {
        'mean_smoothed_auc': float(seed_auc.mean()),
        'standard_error_auc': float(
            seed_auc.std(ddof=1) / np.sqrt(len(seed_auc))),
        'selection_score': float(
            seed_auc.mean() - seed_auc.std(ddof=1) /
            np.sqrt(len(seed_auc))),
        'final_window_reward': float(smoothed[:, -3:].mean()),
        'seed_auc': seed_auc.tolist()
    }


def held_out_summary(curves, smoothing_window):
    rewards = np.stack([curve[:, 1] for curve in curves])
    smoothed_seeds = np.stack([
        smooth(seed_rewards, smoothing_window)
        for seed_rewards in rewards
    ])
    smoothed_mean = smooth(rewards.mean(axis=0), smoothing_window)
    ci95 = (1.96 * smoothed_seeds.std(axis=0, ddof=1) /
            np.sqrt(len(curves)))
    steps = curves[0][:, 0]
    seed_auc = np.asarray([
        np.trapezoid(seed_curve, steps) / steps[-1]
        for seed_curve in smoothed_seeds
    ])
    return {
        'steps': steps.astype(int).tolist(),
        'raw_mean_reward': rewards.mean(axis=0).tolist(),
        'smoothed_mean_reward': smoothed_mean.tolist(),
        'smoothed_ci95_reward': ci95.tolist(),
        'seed_raw_reward': rewards.tolist(),
        'seed_smoothed_reward': smoothed_seeds.tolist(),
        'smoothed_auc': float(seed_auc.mean()),
        'smoothed_auc_ci95': float(
            1.96 * seed_auc.std(ddof=1) / np.sqrt(len(seed_auc))),
        'final_smoothed_reward': float(smoothed_mean[-1]),
        'final_smoothed_ci95': float(ci95[-1]),
        'best_smoothed_reward': float(smoothed_mean.max())
    }


def paired_comparison(results):
    baseline = results['Double DQN']
    proposed = results['Double SOR DQN']
    steps = np.asarray(baseline['steps'])
    baseline_curves = np.asarray(baseline['seed_smoothed_reward'])
    proposed_curves = np.asarray(proposed['seed_smoothed_reward'])
    auc_difference = (
        np.trapezoid(proposed_curves, steps, axis=1) / steps[-1] -
        np.trapezoid(baseline_curves, steps, axis=1) / steps[-1]
    )
    final_difference = proposed_curves[:, -1] - baseline_curves[:, -1]

    def summarize_difference(values):
        return {
            'mean': float(values.mean()),
            'ci95': float(
                1.96 * values.std(ddof=1) / np.sqrt(len(values))),
            'seed_differences': values.tolist(),
            'positive_seed_count': int(np.sum(values > 0))
        }

    return {
        'smoothed_auc_difference': summarize_difference(auc_difference),
        'final_smoothed_reward_difference': summarize_difference(
            final_difference)
    }


def write_csv(path, results):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([
            'algorithm', 'step', 'raw_mean_reward',
            'smoothed_mean_reward', 'smoothed_ci95_reward'
        ])
        for algorithm, result in results.items():
            for row in zip(
                    result['steps'], result['raw_mean_reward'],
                    result['smoothed_mean_reward'],
                    result['smoothed_ci95_reward']):
                writer.writerow([algorithm, *row])


def plot(path, results, factor, smoothing_window):
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    colors = {'Double DQN': 'tab:blue', 'Double SOR DQN': 'tab:orange'}
    for algorithm, result in results.items():
        steps = np.asarray(result['steps'])
        raw = np.asarray(result['raw_mean_reward'])
        mean = np.asarray(result['smoothed_mean_reward'])
        ci95 = np.asarray(result['smoothed_ci95_reward'])
        label = algorithm
        if algorithm == 'Double SOR DQN':
            label += f' (w={factor:g})'
        axis.plot(steps, raw, color=colors[algorithm], alpha=.2,
                  linewidth=1)
        axis.plot(steps, mean, color=colors[algorithm], linewidth=2.4,
                  label=label)
        axis.fill_between(
            steps, np.maximum(0, mean - ci95), np.minimum(500, mean + ci95),
            color=colors[algorithm], alpha=.14)

    axis.axhline(475, color='0.45', linestyle='--', linewidth=1,
                 label='Solved threshold (475)')
    axis.set(
        xlabel='Environment steps', ylabel='Evaluation reward',
        xlim=(0, max(results['Double DQN']['steps'])), ylim=(0, 510),
        title=f'{smoothing_window}-checkpoint centered moving average')
    axis.grid(alpha=.2)
    axis.legend(loc='lower right')
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path,
                        default=Path('cartpole_dsor_smoothed_results'))
    parser.add_argument('--training-steps', type=int, default=25_000)
    parser.add_argument('--tuning-steps', type=int, default=20_000)
    parser.add_argument('--checkpoint', type=int, default=1_000)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--tuning-seeds', type=int, default=5)
    parser.add_argument('--test-seeds', type=int, default=5)
    parser.add_argument('--smoothing-window', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / 'run_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        1.05, 1.0625, 1.075, 1.0875, 1.1, 1.125, 1.15,
        1.2, 1.25, 1.3, 1.35
    ]
    tuning_seed_values = range(100, 100 + args.tuning_seeds)
    tuning_tasks = [
        (cache_dir, 'Double SOR DQN', seed, args.tuning_steps,
         args.checkpoint, args.eval_episodes, factor)
        for factor in candidates for seed in tuning_seed_values
    ]
    collect(tuning_tasks, args.workers)

    tuning = {}
    for factor in candidates:
        curves = [
            load_curve(
                cache_dir, 'Double SOR DQN', seed, args.tuning_steps,
                args.checkpoint, args.eval_episodes, factor)
            for seed in tuning_seed_values
        ]
        tuning[factor] = candidate_summary(
            curves, args.smoothing_window)
    selected_factor = max(
        tuning, key=lambda factor: tuning[factor]['selection_score'])

    test_seed_values = range(2000, 2000 + args.test_seeds)
    test_tasks = [
        (cache_dir, algorithm, seed, args.training_steps,
         args.checkpoint, args.eval_episodes, selected_factor)
        for algorithm in ALGORITHMS for seed in test_seed_values
    ]
    collect(test_tasks, args.workers)

    results = {}
    for algorithm in ALGORITHMS:
        curves = [
            load_curve(
                cache_dir, algorithm, seed, args.training_steps,
                args.checkpoint, args.eval_episodes, selected_factor)
            for seed in test_seed_values
        ]
        results[algorithm] = held_out_summary(
            curves, args.smoothing_window)

    summary = {
        'environment': {
            'name': 'CartPole-v1', 'horizon': 500, 'gamma': .99
        },
        'protocol': {
            **vars(args),
            'output_dir': str(args.output_dir),
            'tuning_seed_range': [100, 99 + args.tuning_seeds],
            'held_out_seed_range': [2000, 1999 + args.test_seeds],
            'selection_rule': ('highest validation smoothed AUC minus one '
                               'standard error'),
            'evaluation': (f'{args.eval_episodes} greedy episodes per '
                           'checkpoint'),
            'smoothing': ('centered moving average applied independently '
                          'within each seed')
        },
        'sor_candidates': candidates,
        'tuning': {str(factor): values
                   for factor, values in tuning.items()},
        'selected_relaxation_factor': selected_factor,
        'results': results,
        'paired_comparison': paired_comparison(results)
    }
    with (args.output_dir / 'summary.json').open('w') as stream:
        json.dump(summary, stream, indent=2)
    write_csv(args.output_dir / 'smoothed_learning_curves.csv', results)
    plot(args.output_dir / 'smoothed_reward_curve.png', results,
         selected_factor, args.smoothing_window)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
