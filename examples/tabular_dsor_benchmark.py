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


class StochasticGridWorld:
    def __init__(self, size=6, gamma=.95, horizon=100, stay_probability=.2,
                 seed=None):
        self.size = size
        self.gamma = gamma
        self.horizon = horizon
        self.stay_probability = stay_probability
        self._rng = np.random.default_rng(seed)
        self._goal = (size - 1, size - 1)
        self.info = MDPInfo(
            Discrete(size * size), Discrete(4), gamma, horizon, .1)

    def reset(self):
        return np.array([0])

    def step(self, state, action):
        row, col = divmod(int(state[0]), self.size)
        if self._rng.random() >= self.stay_probability:
            if action == 0:
                row = max(0, row - 1)
            elif action == 1:
                row = min(self.size - 1, row + 1)
            elif action == 2:
                col = max(0, col - 1)
            else:
                col = min(self.size - 1, col + 1)

        absorbing = (row, col) == self._goal
        reward = 0. if absorbing else -1.
        return np.array([row * self.size + col]), reward, absorbing


def make_agent(name, mdp_info, relaxation_factor=None):
    policy = EpsGreedy(Parameter(0.))
    learning_rate = Parameter(.2)

    if name == 'Q-Learning':
        return QLearning(mdp_info, policy, learning_rate)
    if name == 'Double Q-Learning':
        return DoubleQLearning(mdp_info, policy, learning_rate)
    return DoubleSORQLearning(
        mdp_info, policy, learning_rate, relaxation_factor)


def q_values(agent, state):
    if len(agent.Q) == 1:
        return agent.Q[state, :]
    return .5 * (agent.Q[0][state, :] + agent.Q[1][state, :])


def greedy_action(agent, state, rng):
    q = q_values(agent, state)
    actions = np.flatnonzero(q == np.max(q))
    return int(rng.choice(actions))


def train_episode(agent, env, rng, step_count, exploration_steps):
    state = env.reset()
    for _ in range(env.horizon):
        epsilon = max(.05, 1. - .95 * step_count / exploration_steps)
        if rng.random() < epsilon:
            action = int(rng.integers(4))
        else:
            action = greedy_action(agent, state, rng)

        next_state, reward, absorbing = env.step(state, action)
        agent._update(state, np.array([action]), reward, next_state,
                      absorbing)
        step_count += 1
        state = next_state
        if absorbing:
            break

    return step_count


def evaluate(agent, env, rng, n_episodes):
    lengths = []
    successes = []
    for _ in range(n_episodes):
        state = env.reset()
        for step in range(1, env.horizon + 1):
            action = greedy_action(agent, state, rng)
            state, _, absorbing = env.step(state, action)
            if absorbing:
                break
        lengths.append(step)
        successes.append(absorbing)

    return np.mean(lengths), np.mean(successes)


def run(name, seed, episodes, checkpoint, eval_episodes,
        relaxation_factor=None):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    train_env = StochasticGridWorld(seed=seed)
    eval_env = StochasticGridWorld(seed=seed + 10_000)
    agent = make_agent(name, train_env.info, relaxation_factor)
    exploration_steps = episodes * 30
    step_count = 0
    curve = []

    for episode in range(episodes + 1):
        if episode % checkpoint == 0:
            length, success = evaluate(
                agent, eval_env, rng, eval_episodes)
            curve.append((episode, length, success))
        if episode < episodes:
            step_count = train_episode(
                agent, train_env, rng, step_count, exploration_steps)

    return np.asarray(curve)


def tune_relaxation_factor(candidates, seeds, **run_params):
    scores = {}
    for factor in candidates:
        curves = [run('Double SOR Q-Learning', seed,
                      relaxation_factor=factor, **run_params)
                  for seed in seeds]
        scores[factor] = np.mean([np.trapezoid(c[:, 1], c[:, 0])
                                  for c in curves])
    return scores


def summarize(curves):
    values = np.stack([curve[:, 1] for curve in curves])
    success = np.stack([curve[:, 2] for curve in curves])
    return {
        'episodes': curves[0][:, 0].tolist(),
        'mean_steps': values.mean(0).tolist(),
        'ci95_steps': (1.96 * values.std(0, ddof=1) /
                       np.sqrt(len(values))).tolist(),
        'mean_success': success.mean(0).tolist(),
        'auc_steps': float(np.mean([
            np.trapezoid(curve[:, 1], curve[:, 0]) for curve in curves
        ])),
        'final_steps': float(values[:, -3:].mean()),
        'final_success': float(success[:, -3:].mean())
    }


def write_csv(path, results):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([
            'algorithm', 'episode', 'mean_steps', 'ci95_steps',
            'mean_success'
        ])
        for algorithm, result in results.items():
            for row in zip(result['episodes'], result['mean_steps'],
                           result['ci95_steps'], result['mean_success']):
                writer.writerow([algorithm, *row])


def plot_results(path, tuning_scores, selected_factor, results):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    factors = list(tuning_scores)
    axes[0].plot(factors, [tuning_scores[w] for w in factors], marker='o')
    axes[0].axvline(selected_factor, color='tab:red', linestyle='--',
                    label=f'selected w={selected_factor:.2f}')
    axes[0].set(xlabel='Relaxation factor w',
                ylabel='Validation area under steps curve',
                title='SOR-factor tuning (lower is better)')
    axes[0].legend()

    for algorithm, result in results.items():
        x = np.asarray(result['episodes'])
        y = np.asarray(result['mean_steps'])
        ci = np.asarray(result['ci95_steps'])
        axes[1].plot(x, y, label=algorithm)
        axes[1].fill_between(x, y - ci, y + ci, alpha=.15)
    axes[1].set(xlabel='Training episodes',
                ylabel='Greedy steps to goal',
                title='Held-out learning curves (lower is better)')
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path,
                        default=Path('benchmark_results'))
    parser.add_argument('--episodes', type=int, default=180)
    parser.add_argument('--checkpoint', type=int, default=10)
    parser.add_argument('--eval-episodes', type=int, default=30)
    parser.add_argument('--tuning-seeds', type=int, default=12)
    parser.add_argument('--test-seeds', type=int, default=40)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_params = dict(
        episodes=args.episodes,
        checkpoint=args.checkpoint,
        eval_episodes=args.eval_episodes
    )
    candidates = [1., 1.05, 1.1, 1.15, 1.2, 1.23]
    tuning_scores = tune_relaxation_factor(
        candidates, range(args.tuning_seeds), **run_params)
    selected_factor = min(tuning_scores, key=tuning_scores.get)

    algorithms = ['Q-Learning', 'Double Q-Learning',
                  'Double SOR Q-Learning']
    results = {}
    for algorithm in algorithms:
        curves = [run(
            algorithm, seed, relaxation_factor=selected_factor,
            **run_params
        ) for seed in range(1000, 1000 + args.test_seeds)]
        results[algorithm] = summarize(curves)

    output = {
        'environment': {
            'name': '6x6 stochastic grid world',
            'gamma': .95,
            'stay_probability': .2,
            'horizon': 100,
            'reward': '-1 per non-terminal step, 0 at goal'
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
    plot_results(args.output_dir / 'comparison.png', tuning_scores,
                 selected_factor, results)

    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
