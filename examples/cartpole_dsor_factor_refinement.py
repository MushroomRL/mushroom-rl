import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cartpole_dsor_dqn_smoothed_benchmark import (
    collect, held_out_summary, load_curve, smooth
)


def seed_metrics(curves, smoothing_window):
    steps = curves[0][:, 0]
    smoothed = np.stack([
        smooth(curve[:, 1], smoothing_window) for curve in curves
    ])
    auc = np.trapezoid(smoothed, steps, axis=1) / steps[-1]
    late = smoothed[:, -5:].mean(axis=1)
    return auc, late


def paired_factor_summary(proposed_curves, baseline_curves,
                          smoothing_window):
    proposed_auc, proposed_late = seed_metrics(
        proposed_curves, smoothing_window)
    baseline_auc, baseline_late = seed_metrics(
        baseline_curves, smoothing_window)
    auc_difference = proposed_auc - baseline_auc
    late_difference = proposed_late - baseline_late
    combined = .5 * auc_difference + .5 * late_difference
    standard_error = combined.std(ddof=1) / np.sqrt(len(combined))
    return {
        'mean_auc_difference': float(auc_difference.mean()),
        'mean_late_reward_difference': float(late_difference.mean()),
        'mean_combined_difference': float(combined.mean()),
        'standard_error_combined': float(standard_error),
        'conservative_difference': float(
            combined.mean() - standard_error),
        'positive_auc_seed_count': int(np.sum(auc_difference > 0)),
        'positive_late_seed_count': int(np.sum(late_difference > 0)),
        'seed_auc_differences': auc_difference.tolist(),
        'seed_late_differences': late_difference.tolist()
    }


def paired_confirmation(results):
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


def write_refinement_csv(path, summaries):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([
            'relaxation_factor', 'mean_auc_difference',
            'mean_late_reward_difference', 'mean_combined_difference',
            'standard_error_combined', 'conservative_difference'
        ])
        for factor, result in summaries.items():
            writer.writerow([
                factor, result['mean_auc_difference'],
                result['mean_late_reward_difference'],
                result['mean_combined_difference'],
                result['standard_error_combined'],
                result['conservative_difference']
            ])


def plot(path, summaries, selected_factor, results, smoothing_window):
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    factors = np.asarray(list(summaries))
    scores = np.asarray([
        summaries[factor]['conservative_difference']
        for factor in factors
    ])
    axes[0].axhline(0, color='0.45', linewidth=1)
    axes[0].plot(factors, scores, marker='o', linewidth=1.8)
    axes[0].axvline(
        selected_factor, color='tab:orange', linestyle='--',
        label=f'selected w={selected_factor:g}')
    axes[0].set(
        xlabel='Relaxation factor w',
        ylabel='Conservative paired improvement',
        title='Ten-seed factor refinement')
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
        title=(f'Fresh confirmation; {smoothing_window}-checkpoint '
               'moving average'))
    axes[1].grid(alpha=.2)
    axes[1].legend(loc='lower right')
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path,
                        default=Path(
                            'cartpole_dsor_factor_refinement_results'))
    parser.add_argument('--cache-dir', type=Path,
                        default=Path(
                            'cartpole_dsor_factor_sweep_results/run_cache'))
    parser.add_argument('--training-steps', type=int, default=25_000)
    parser.add_argument('--checkpoint', type=int, default=1_000)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--smoothing-window', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    factors = [1.005, 1.01, 1.02, 1.03, 1.04]
    refinement_seeds = list(range(2000, 2005)) + list(range(4000, 4005))
    refinement_tasks = [
        (args.cache_dir, algorithm, seed, args.training_steps,
         args.checkpoint, args.eval_episodes, factor)
        for algorithm, factor in (
            [('Double DQN', 1.)] +
            [('Double SOR DQN', factor) for factor in factors])
        for seed in refinement_seeds
    ]
    collect(refinement_tasks, args.workers)

    baseline_curves = [
        load_curve(
            args.cache_dir, 'Double DQN', seed, args.training_steps,
            args.checkpoint, args.eval_episodes, 1.)
        for seed in refinement_seeds
    ]
    summaries = {}
    for factor in factors:
        proposed_curves = [
            load_curve(
                args.cache_dir, 'Double SOR DQN', seed,
                args.training_steps, args.checkpoint,
                args.eval_episodes, factor)
            for seed in refinement_seeds
        ]
        summaries[factor] = paired_factor_summary(
            proposed_curves, baseline_curves, args.smoothing_window)
    selected_factor = max(
        summaries,
        key=lambda factor: summaries[factor]['conservative_difference'])

    confirmation_seeds = range(6000, 6005)
    confirmation_tasks = [
        (args.cache_dir, algorithm, seed, args.training_steps,
         args.checkpoint, args.eval_episodes, selected_factor)
        for algorithm in ('Double DQN', 'Double SOR DQN')
        for seed in confirmation_seeds
    ]
    collect(confirmation_tasks, args.workers)

    results = {}
    for algorithm in ('Double DQN', 'Double SOR DQN'):
        curves = [
            load_curve(
                args.cache_dir, algorithm, seed, args.training_steps,
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
            'cache_dir': str(args.cache_dir),
            'refinement_seed_ranges': [[2000, 2004], [4000, 4004]],
            'confirmation_seed_range': [6000, 6004],
            'selection_rule': (
                'largest paired conservative difference using equal '
                'weight on smoothed AUC and final-five-checkpoint reward')
        },
        'factors': factors,
        'refinement_results': {
            str(factor): result for factor, result in summaries.items()
        },
        'selected_relaxation_factor': selected_factor,
        'confirmation_results': results,
        'paired_confirmation': paired_confirmation(results)
    }
    with (args.output_dir / 'summary.json').open('w') as stream:
        json.dump(summary, stream, indent=2)
    write_refinement_csv(
        args.output_dir / 'factor_refinement.csv', summaries)
    plot(
        args.output_dir / 'refinement_and_confirmation.png',
        summaries, selected_factor, results, args.smoothing_window)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
