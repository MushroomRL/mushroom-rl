import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SOURCE_FILES = {
    'DQN': 'ProbLeft-Q',
    'SOR DQN (w=1.3)': 'ProbLeft-SORQ',
    'Double DQN': 'ProbLeft-D-Q-average',
    'Double SOR DQN (w=1.3)': 'ProbLeft-SORDQ-average'
}


def load_results(source_dir):
    results = {}
    for algorithm, filename in SOURCE_FILES.items():
        values = np.load(source_dir / filename, allow_pickle=False)
        if values.shape != (400, 1000):
            raise ValueError(
                f'{filename} has shape {values.shape}; expected (400, 1000)')
        results[algorithm] = values
    return results


def summarize(values):
    mean = values.mean(axis=1)
    ci95 = 1.96 * values.std(axis=1, ddof=1) / np.sqrt(values.shape[1])
    return {
        'episodes': np.arange(1, len(mean) + 1).tolist(),
        'mean_probability_left': mean.tolist(),
        'ci95_probability_left': ci95.tolist(),
        'final_probability_left': float(mean[-1]),
        'final_ci95': float(ci95[-1]),
        'last_50_episode_mean': float(mean[-50:].mean()),
        'curve_auc': float(np.trapezoid(mean) / (len(mean) - 1))
    }


def write_csv(path, summaries):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([
            'algorithm', 'episode', 'mean_probability_left',
            'ci95_probability_left'
        ])
        for algorithm, result in summaries.items():
            rows = zip(
                result['episodes'], result['mean_probability_left'],
                result['ci95_probability_left'])
            for row in rows:
                writer.writerow([algorithm, *row])


def plot(path, summaries):
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    colors = {
        'DQN': 'tab:blue',
        'SOR DQN (w=1.3)': 'tab:green',
        'Double DQN': 'tab:purple',
        'Double SOR DQN (w=1.3)': 'tab:orange'
    }
    styles = {
        'DQN': '--',
        'SOR DQN (w=1.3)': '-.',
        'Double DQN': ':',
        'Double SOR DQN (w=1.3)': '-'
    }
    for algorithm, result in summaries.items():
        episodes = np.asarray(result['episodes'])
        mean = np.asarray(result['mean_probability_left'])
        ci95 = np.asarray(result['ci95_probability_left'])
        axis.plot(
            episodes, mean, color=colors[algorithm],
            linestyle=styles[algorithm], linewidth=2,
            label=algorithm)
        axis.fill_between(
            episodes, np.maximum(0, mean - ci95),
            np.minimum(1, mean + ci95), color=colors[algorithm],
            alpha=.08)

    axis.set(
        xlabel='Training episode',
        ylabel='Probability of selecting the biased left action',
        xlim=(1, 400), ylim=(0, .55))
    axis.grid(alpha=.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source-dir', type=Path,
        default=Path('neural_maximization_bias_results/source_arrays'))
    parser.add_argument(
        '--output-dir', type=Path,
        default=Path('neural_maximization_bias_results'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_results = load_results(args.source_dir)
    summaries = {
        algorithm: summarize(values)
        for algorithm, values in raw_results.items()
    }
    double_dqn = summaries['Double DQN']
    double_sor = summaries['Double SOR DQN (w=1.3)']
    comparison = {
        'final_probability_reduction': float(
            1. - double_sor['final_probability_left'] /
            double_dqn['final_probability_left']),
        'last_50_probability_reduction': float(
            1. - double_sor['last_50_episode_mean'] /
            double_dqn['last_50_episode_mean']),
        'curve_auc_reduction': float(
            1. - double_sor['curve_auc'] / double_dqn['curve_auc'])
    }
    output = {
        'source': {
            'repository': (
                'https://github.com/shreyassr123/'
                'Double-SOR-Q-Learning'),
            'commit': '8a26c4ca37f2447b74d10572e57d5e4ef16a57a4',
            'source_file': (
                'Deep RL Version/Maximization Bias/nn_biasexample.py'),
            'result_shape': [400, 1000]
        },
        'environment': {
            'description': 'neural-network maximization-bias example',
            'states': 10**9 + 2,
            'state_dimension': 1,
            'actions': 2,
            'biased_transition_reward': 'Normal(-0.1, 1)',
            'metric': (
                'probability of selecting the left action; lower is better')
        },
        'hyperparameters': {
            'network': [1, 4, 8, 2],
            'activation': 'ReLU',
            'optimizer': 'SGD',
            'learning_rate': '10 / (episode + 100)',
            'gamma': .999,
            'epsilon': .1,
            'episodes': 400,
            'iterations': 1000,
            'relaxation_factor': 1.3,
            'double_policy_evaluation': 'mean of the two estimators'
        },
        'results': summaries,
        'double_sor_vs_double_dqn': comparison
    }
    with (args.output_dir / 'summary.json').open('w') as stream:
        json.dump(output, stream, indent=2)
    write_csv(args.output_dir / 'learning_curves.csv', summaries)
    plot(args.output_dir / 'comparison.png', summaries)
    print(json.dumps({
        'results': {
            algorithm: {
                'final_probability_left': result[
                    'final_probability_left'],
                'final_ci95': result['final_ci95'],
                'last_50_episode_mean': result[
                    'last_50_episode_mean'],
                'curve_auc': result['curve_auc']
            }
            for algorithm, result in summaries.items()
        },
        'double_sor_vs_double_dqn': comparison
    }, indent=2))


if __name__ == '__main__':
    main()
