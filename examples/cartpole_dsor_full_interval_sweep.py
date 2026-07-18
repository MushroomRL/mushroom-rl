import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cartpole_dsor_dqn_smoothed_benchmark import (
    collect, held_out_summary, load_curve, smooth
)


def curve_metrics(curves, smoothing_window):
    steps = curves[0][:, 0]
    smoothed = np.stack([
        smooth(curve[:, 1], smoothing_window) for curve in curves
    ])
    auc = np.trapezoid(smoothed, steps, axis=1) / steps[-1]
    late = smoothed[:, -5:].mean(axis=1)
    return auc, late


def coarse_summary(curves, smoothing_window):
    auc, late = curve_metrics(curves, smoothing_window)
    return {
        'mean_smoothed_auc': float(auc.mean()),
        'standard_error_auc': float(
            auc.std(ddof=1) / np.sqrt(len(auc))),
        'mean_late_reward': float(late.mean()),
        'seed_auc': auc.tolist(),
        'seed_late_reward': late.tolist()
    }


def paired_summary(proposed_curves, baseline_curves, smoothing_window):
    proposed_auc, proposed_late = curve_metrics(
        proposed_curves, smoothing_window)
    baseline_auc, baseline_late = curve_metrics(
        baseline_curves, smoothing_window)
    auc_difference = proposed_auc - baseline_auc
    late_difference = proposed_late - baseline_late
    combined = .75 * auc_difference + .25 * late_difference
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


def write_csv(path, results, fields):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['relaxation_factor', *fields])
        for factor, result in results.items():
            writer.writerow([factor, *[result[field] for field in fields]])


def plot(path, coarse, refined, selected_factor, confirmation):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    coarse_factors = np.asarray(list(coarse))
    coarse_auc = np.asarray([
        coarse[factor]['mean_smoothed_auc'] for factor in coarse_factors
    ])
    axes[0].semilogx(
        coarse_factors, coarse_auc, marker='o', linewidth=1.6)
    axes[0].set(
        xlabel='Relaxation factor w (log scale)',
        ylabel='10k-step smoothed AUC',
        title='Full admissible-interval screen')
    axes[0].grid(alpha=.2)

    refined_factors = np.asarray(sorted(refined))
    refined_scores = np.asarray([
        refined[factor]['conservative_difference']
        for factor in refined_factors
    ])
    axes[1].axhline(0, color='0.45', linewidth=1)
    axes[1].plot(
        refined_factors, refined_scores, marker='o', linewidth=1.8)
    axes[1].axvline(
        selected_factor, color='tab:orange', linestyle='--',
        label=f'selected w={selected_factor:g}')
    axes[1].set(
        xlabel='Relaxation factor w',
        ylabel='Conservative paired improvement',
        title='Long-run refinement')
    axes[1].grid(alpha=.2)
    axes[1].legend()

    colors = {'Double DQN': 'tab:blue', 'Double SOR DQN': 'tab:orange'}
    for algorithm, result in confirmation.items():
        steps = np.asarray(result['steps'])
        mean = np.asarray(result['smoothed_mean_reward'])
        ci95 = np.asarray(result['smoothed_ci95_reward'])
        label = algorithm
        if algorithm == 'Double SOR DQN':
            label += f' (w={selected_factor:g})'
        axes[2].plot(
            steps, mean, color=colors[algorithm], linewidth=2.3,
            label=label)
        axes[2].fill_between(
            steps, np.maximum(0, mean - ci95),
            np.minimum(500, mean + ci95),
            color=colors[algorithm], alpha=.14)
    axes[2].set(
        xlabel='Environment steps', ylabel='Evaluation reward',
        xlim=(0, max(confirmation['Double DQN']['steps'])),
        ylim=(0, 510), title='Fresh-seed confirmation')
    axes[2].grid(alpha=.2)
    axes[2].legend(loc='lower right')
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path,
                        default=Path(
                            'cartpole_dsor_full_interval_results'))
    parser.add_argument('--coarse-steps', type=int, default=10_000)
    parser.add_argument('--training-steps', type=int, default=25_000)
    parser.add_argument('--checkpoint', type=int, default=1_000)
    parser.add_argument('--coarse-eval-episodes', type=int, default=5)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--smoothing-window', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / 'run_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    gamma = .99
    theoretical_upper_bound = 1. / (1. - gamma)
    factors = [
        1.001, 1.005, 1.01, 1.02, 1.04, 1.08, 1.15, 1.3,
        1.5, 2., 3., 5., 8., 12., 20., 35., 50., 70., 90., 99.
    ]
    coarse_seeds = range(7000, 7003)
    coarse_tasks = [
        (cache_dir, 'Double SOR DQN', seed, args.coarse_steps,
         args.checkpoint, args.coarse_eval_episodes, factor)
        for factor in factors for seed in coarse_seeds
    ]
    collect(coarse_tasks, args.workers)

    coarse_results = {}
    for factor in factors:
        curves = [
            load_curve(
                cache_dir, 'Double SOR DQN', seed,
                args.coarse_steps, args.checkpoint,
                args.coarse_eval_episodes, factor)
            for seed in coarse_seeds
        ]
        coarse_results[factor] = coarse_summary(
            curves, args.smoothing_window)

    refined_factors = sorted(
        coarse_results,
        key=lambda factor: coarse_results[factor]['mean_smoothed_auc'],
        reverse=True)[:5]
    refinement_seeds = range(8000, 8005)
    refinement_tasks = [
        (cache_dir, algorithm, seed, args.training_steps,
         args.checkpoint, args.eval_episodes, factor)
        for algorithm, factor in (
            [('Double DQN', 1.)] +
            [('Double SOR DQN', factor)
             for factor in refined_factors])
        for seed in refinement_seeds
    ]
    collect(refinement_tasks, args.workers)

    baseline_curves = [
        load_curve(
            cache_dir, 'Double DQN', seed, args.training_steps,
            args.checkpoint, args.eval_episodes, 1.)
        for seed in refinement_seeds
    ]
    refined_results = {}
    for factor in refined_factors:
        proposed_curves = [
            load_curve(
                cache_dir, 'Double SOR DQN', seed,
                args.training_steps, args.checkpoint,
                args.eval_episodes, factor)
            for seed in refinement_seeds
        ]
        refined_results[factor] = paired_summary(
            proposed_curves, baseline_curves, args.smoothing_window)
    selected_factor = max(
        refined_results,
        key=lambda factor: refined_results[factor][
            'conservative_difference'])

    confirmation_seeds = range(9000, 9005)
    confirmation_tasks = [
        (cache_dir, algorithm, seed, args.training_steps,
         args.checkpoint, args.eval_episodes, selected_factor)
        for algorithm in ('Double DQN', 'Double SOR DQN')
        for seed in confirmation_seeds
    ]
    collect(confirmation_tasks, args.workers)

    confirmation = {}
    for algorithm in ('Double DQN', 'Double SOR DQN'):
        curves = [
            load_curve(
                cache_dir, algorithm, seed, args.training_steps,
                args.checkpoint, args.eval_episodes, selected_factor)
            for seed in confirmation_seeds
        ]
        confirmation[algorithm] = held_out_summary(
            curves, args.smoothing_window)

    summary = {
        'environment': {
            'name': 'CartPole-v1', 'gamma': gamma, 'horizon': 500
        },
        'theoretical_interval': {
            'lower_bound': 1.,
            'upper_bound': theoretical_upper_bound,
            'upper_bound_excluded': True
        },
        'protocol': {
            **vars(args),
            'output_dir': str(args.output_dir),
            'coarse_seed_range': [7000, 7002],
            'refinement_seed_range': [8000, 8004],
            'confirmation_seed_range': [9000, 9004],
            'selection_rule': (
                'top five by full-interval screen, then largest paired '
                'conservative improvement over Double DQN')
        },
        'coarse_factors': factors,
        'coarse_results': {
            str(factor): result
            for factor, result in coarse_results.items()
        },
        'refined_factors': refined_factors,
        'refined_results': {
            str(factor): result
            for factor, result in refined_results.items()
        },
        'selected_relaxation_factor': selected_factor,
        'confirmation_results': confirmation,
        'paired_confirmation': paired_confirmation(confirmation)
    }
    with (args.output_dir / 'summary.json').open('w') as stream:
        json.dump(summary, stream, indent=2)
    write_csv(
        args.output_dir / 'full_interval_screen.csv', coarse_results,
        ['mean_smoothed_auc', 'standard_error_auc',
         'mean_late_reward'])
    write_csv(
        args.output_dir / 'long_run_refinement.csv', refined_results,
        ['mean_auc_difference', 'mean_late_reward_difference',
         'conservative_difference'])
    plot(
        args.output_dir / 'full_interval_sweep.png', coarse_results,
        refined_results, selected_factor, confirmation)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
