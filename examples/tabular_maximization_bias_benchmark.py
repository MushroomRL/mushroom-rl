import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mushroom_rl.algorithms.value import QLearning, DoubleQLearning, \
    DoubleSORQLearning
from mushroom_rl.core import MDPInfo
from mushroom_rl.core.spaces import Discrete
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter


class NoisySingleStateMDP:
    def __init__(self, n_actions=10, gamma=.9, seed=None):
        self.n_actions = n_actions
        self.means = np.full(n_actions, -.1)
        self.means[0] = 0.
        self.standard_deviations = np.ones(n_actions)
        self.standard_deviations[0] = .05
        self._rng = np.random.default_rng(seed)
        self.info = MDPInfo(
            Discrete(1), Discrete(n_actions), gamma, np.inf, .1)

    def sample_reward(self, action):
        return self._rng.normal(
            self.means[action], self.standard_deviations[action])


def make_agent(name, mdp_info, relaxation_factor=None):
    policy = EpsGreedy(Parameter(0.))
    learning_rate = Parameter(.05)
    if name == 'Q-Learning':
        return QLearning(mdp_info, policy, learning_rate)
    if name == 'Double Q-Learning':
        return DoubleQLearning(mdp_info, policy, learning_rate)
    return DoubleSORQLearning(
        mdp_info, policy, learning_rate, relaxation_factor)


def values(agent, state):
    if len(agent.Q) == 1:
        return agent.Q[state, :]
    return .5 * (agent.Q[0][state, :] + agent.Q[1][state, :])


def greedy_action(agent, state, rng):
    q = values(agent, state)
    actions = np.flatnonzero(q == np.max(q))
    return int(rng.choice(actions))


def run(name, seed, n_steps, checkpoint, relaxation_factor=None):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    mdp = NoisySingleStateMDP(seed=seed + 10_000)
    agent = make_agent(name, mdp.info, relaxation_factor)
    state = np.array([0])
    curve = []

    for step in range(n_steps + 1):
        if step % checkpoint == 0:
            action = greedy_action(agent, state, rng)
            curve.append((step, mdp.means[action], action == 0))
        if step == n_steps:
            break

        epsilon = max(.02, 1. - .98 * step / (n_steps // 2))
        if rng.random() < epsilon:
            action = int(rng.integers(mdp.n_actions))
        else:
            action = greedy_action(agent, state, rng)
        reward = mdp.sample_reward(action)
        agent._update(state, np.array([action]), reward, state, False)

    return np.asarray(curve)


def tune(candidates, seeds, **run_params):
    scores = {}
    for factor in candidates:
        curves = [run('Double SOR Q-Learning', seed,
                      relaxation_factor=factor, **run_params)
                  for seed in seeds]
        scores[factor] = float(np.mean([
            np.trapezoid(-curve[:, 1], curve[:, 0]) for curve in curves
        ]))
    return scores


def summarize(curves):
    expected_rewards = np.stack([curve[:, 1] for curve in curves])
    optimal = np.stack([curve[:, 2] for curve in curves])
    return {
        'steps': curves[0][:, 0].tolist(),
        'mean_expected_reward': expected_rewards.mean(0).tolist(),
        'ci95_expected_reward': (
            1.96 * expected_rewards.std(0, ddof=1) /
            np.sqrt(len(curves))).tolist(),
        'optimal_action_rate': optimal.mean(0).tolist(),
        'auc_regret': float(np.mean([
            np.trapezoid(-curve[:, 1], curve[:, 0]) for curve in curves
        ])),
        'final_optimal_action_rate': float(optimal[:, -5:].mean()),
        'final_expected_reward': float(expected_rewards[:, -5:].mean())
    }


def write_csv(path, results):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([
            'algorithm', 'step', 'mean_expected_reward',
            'ci95_expected_reward', 'optimal_action_rate'
        ])
        for algorithm, result in results.items():
            rows = zip(
                result['steps'], result['mean_expected_reward'],
                result['ci95_expected_reward'],
                result['optimal_action_rate'])
            for row in rows:
                writer.writerow([algorithm, *row])


def plot(path, tuning_scores, selected_factor, results):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    factors = list(tuning_scores)
    axes[0].plot(factors, [tuning_scores[w] for w in factors], marker='o')
    axes[0].axvline(selected_factor, color='tab:red', linestyle='--',
                    label=f'selected w={selected_factor:g}')
    axes[0].set(xlabel='Relaxation factor w',
                ylabel='Validation cumulative regret',
                title='SOR-factor tuning (lower is better)')
    axes[0].legend()

    for algorithm, result in results.items():
        axes[1].plot(result['steps'], result['optimal_action_rate'],
                     label=algorithm)
    axes[1].set(xlabel='Training steps', ylabel='Optimal-action rate',
                ylim=(0., 1.02),
                title='Held-out maximization-bias benchmark')
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path,
                        default=Path('maximization_bias_results'))
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--checkpoint', type=int, default=100)
    parser.add_argument('--tuning-seeds', type=int, default=40)
    parser.add_argument('--test-seeds', type=int, default=200)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_params = dict(n_steps=args.steps, checkpoint=args.checkpoint)
    candidates = [1., 1.25, 1.5, 2., 3., 4., 5., 6., 7., 8., 9.,
                  9.5]
    tuning_scores = tune(
        candidates, range(args.tuning_seeds), **run_params)
    selected_factor = min(tuning_scores, key=tuning_scores.get)

    results = {}
    algorithms = ['Q-Learning', 'Double Q-Learning',
                  'Double SOR Q-Learning']
    for algorithm in algorithms:
        curves = [run(
            algorithm, seed, relaxation_factor=selected_factor,
            **run_params
        ) for seed in range(1000, 1000 + args.test_seeds)]
        results[algorithm] = summarize(curves)

    output = {
        'environment': {
            'name': 'single-state maximization-bias benchmark',
            'gamma': .9,
            'actions': 10,
            'optimal_action': {
                'mean_reward': 0., 'standard_deviation': .05
            },
            'suboptimal_actions': {
                'count': 9, 'mean_reward': -.1,
                'standard_deviation': 1.
            }
        },
        'protocol': {
            **vars(args),
            'output_dir': str(args.output_dir),
            'tuning_seed_range': [0, args.tuning_seeds - 1],
            'held_out_seed_range': [1000, 999 + args.test_seeds]
        },
        'tuning_scores': {str(w): score
                          for w, score in tuning_scores.items()},
        'selected_relaxation_factor': selected_factor,
        'results': results
    }
    with (args.output_dir / 'summary.json').open('w') as stream:
        json.dump(output, stream, indent=2)
    write_csv(args.output_dir / 'learning_curves.csv', results)
    plot(args.output_dir / 'comparison.png', tuning_scores,
         selected_factor, results)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
