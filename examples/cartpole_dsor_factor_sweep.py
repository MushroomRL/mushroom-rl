import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cartpole_dsor_dqn_smoothed_benchmark import (
    collect, held_out_summary, load_curve, smooth
)


def factor_summary(curves, smoothing_window):
    steps = curves[0][:, 0]
    smoothed = np.stack([
        smooth(curve[:, 1], smoothing_window) for curve in curves
    ])
    seed_auc = np.trapezoid(smoothed, steps, axis=1) / steps[-1]
    seed_late_reward = smoothed[:, -5:].mean(axis=1)
    seed_score = .75 * seed_auc + .25 * seed_late_reward
    standard_error = seed_score.std(ddof=1) / np.sqrt(len(seed_score))
    return {
        'mean_smoothed_auc': float(seed_auc.mean()),
        'mean_late_reward': float(seed_late_reward.mean()),
        'mean_score': float(seed_score.mean()),
        'standard_error_score': float(standard_error),
        'conservative_score': float(seed_score.mean() - standard_error),
        'seed_smoothed_auc': seed_auc.tolist(),
        'seed_late_reward': seed_late_reward.tolist()
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
    late_difference = (
        proposed_curves[:, -5:].mean(axis=1) -
        baseline_curves[:, -5:].mean(axis=1)
    )

    def describe(values):
        return {
            'mean': float(values.mean()),
            'ci95': float(
                1.96 * values.std(ddof=1) / np.sqrt(len(values))),
            'positive_seed_count': int(np.sum(values > 0)),
            'seed_differences': values.tolist()
        }

    return {
        'smoothed_auc_difference': describe(auc_difference),
        'late_reward_difference': describe(late_difference)
    }


def write_sweep_csv(path, factor_results):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([
            'relaxation_factor', 'mean_smoothed_auc', 'mean_late_reward',
            'mean_score', 'standard_error_score', 'conservative_score'
        ])
        for factor, result in factor_results.items():
            writer.writerow([
                factor, result['mean_smoothed_auc'],
                result['mean_late_reward'], result['mean_score'],
                result['standard_error_score'],
                result['conservative_score']
            ])


def write_curves_csv(path, results):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([
            'algorithm', 'step', 'raw_mean_reward',
            'smoothed_mean_reward', 'smoothed_ci95_reward'
        ])
        for algorithm, result in results.items():
            rows = zip(
                result['steps'], result['raw_mean_reward'],
                result['smoothed_mean_reward'],
                result['smoothed_ci95_reward'])
            for row in rows:
                writer.writerow([algorithm, *row])


def plot(path, factor_results, selected_factor, results, smoothing_window):
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    factors = np.asarray(list(factor_results))
    scores = np.asarray([
        factor_results[factor]['conservative_score']
        for factor in factors
    ])
    axes[0].plot(factors, scores, marker='o', linewidth=1.8)
    axes[0].axvline(
        selected_factor, color='tab:orange', linestyle='--',
        label=f'selected w={selected_factor:g}')
    axes[0].set(
        xlabel='Relaxation factor w',
        ylabel='Conservative screening score',
        title='SOR-factor sweep')
    axes[0].grid(alpha=.2)
    axes[0].legend()

    colors = {'Double DQN': 'tab:blue', 'Double SOR DQN': 'tab:orange'}
    for algorithm, result in results.items():
        steps = np.asarray(result['steps'])
        mean = np.asarray(result['smoothed_mean_reward'])
        ci95 = np.asarray(result['smoothed_ci95_reward'])
        label = algorithm
        if algorithm == 'Double SOR DQN':
            label += f' (w={selected_factor:g})'
        axes[1].plot(
            steps, mean, color=colors[algorithm], linewidth=2.3,
            label=label)
        axes[1].fill_between(
            steps, np.maximum(0, mean - ci95),
            np.minimum(500, mean + ci95),
            color=colors[algorithm], alpha=.14)
    axes[1].set(
        xlabel='Environment steps', ylabel='Evaluation reward',
        xlim=(0, max(results['Double DQN']['steps'])), ylim=(0, 510),
        title=(f'Fresh-seed confirmation; {smoothing_window}-checkpoint '
               'moving average'))
    axes[1].grid(alpha=.2)
    axes[1].legend(loc='lower right')
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path,
                        default=Path('cartpole_dsor_factor_sweep_results'))
    parser.add_argument('--training-steps', type=int, default=25_000)
    parser.add_argument('--checkpoint', type=int, default=1_000)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--screening-seeds', type=int, default=5)
    parser.add_argument('--confirmation-seeds', type=int, default=5)
    parser.add_argument('--smoothing-window', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / 'run_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    factors = [
        1.005, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07,
        1.075, 1.08, 1.09, 1.1, 1.125, 1.15, 1.2
    ]
    screening_seeds = range(2000, 2000 + args.screening_seeds)
    screening_tasks = [
        (cache_dir, 'Double SOR DQN', seed, args.training_steps,
         args.checkpoint, args.eval_episodes, factor)
        for factor in factors for seed in screening_seeds
    ]
    collect(screening_tasks, args.workers)

    factor_results = {}
    for factor in factors:
        curves = [
            load_curve(
                cache_dir, 'Double SOR DQN', seed, args.training_steps,
                args.checkpoint, args.eval_episodes, factor)
            for seed in screening_seeds
        ]
        factor_results[factor] = factor_summary(
            curves, args.smoothing_window)
    selected_factor = max(
        factor_results,
        key=lambda factor: factor_results[factor]['conservative_score'])

    confirmation_seeds = range(
        4000, 4000 + args.confirmation_seeds)
    confirmation_tasks = [
        (cache_dir, algorithm, seed, args.training_steps,
         args.checkpoint, args.eval_episodes, selected_factor)
        for algorithm in ('Double DQN', 'Double SOR DQN')
        for seed in confirmation_seeds
    ]
    collect(confirmation_tasks, args.workers)

    results = {}
    for algorithm in ('Double DQN', 'Double SOR DQN'):
        curves = [
            load_curve(
                cache_dir, algorithm, seed, args.training_steps,
                args.checkpoint, args.eval_episodes, selected_factor)
            for seed in confirmation_seeds
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
            'screening_seed_range': [
                2000, 1999 + args.screening_seeds],
            'confirmation_seed_range': [
                4000, 3999 + args.confirmation_seeds],
            'selection_score': (
                '0.75 * smoothed AUC + 0.25 * late reward - one '
                'standard error'),
            'late_reward': 'mean of final five smoothed checkpoints'
        },
        'factors': factors,
        'factor_results': {
            str(factor): result
            for factor, result in factor_results.items()
        },
        'selected_relaxation_factor': selected_factor,
        'confirmation_results': results,
        'paired_confirmation': paired_comparison(results)
    }
    with (args.output_dir / 'summary.json').open('w') as stream:
        json.dump(summary, stream, indent=2)
    write_sweep_csv(
        args.output_dir / 'factor_sweep.csv', factor_results)
    write_curves_csv(
        args.output_dir / 'confirmation_curves.csv', results)
    plot(
        args.output_dir / 'factor_sweep_and_confirmation.png',
        factor_results, selected_factor, results,
        args.smoothing_window)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
