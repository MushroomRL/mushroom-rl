"""
This script aims to replicate the experiments on the Taxi MDP as presented in:
"An Alternative Softmax Operator for Reinforcement Learning", Asadi K. et al. 2017.

"""
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

from mushroom_rl.algorithms.value import SARSA
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Taxi
from mushroom_rl.policy import Boltzmann, EpsGreedy, Mellowmax
from mushroom_rl.utils.callbacks import CollectDataset
from mushroom_rl.rl_utils.parameters import Parameter
from mushroom_rl.utils.experiments import get_log_dir

matplotlib.use('Agg')


def plot_results(curves, symbols, title):
    """
    Draw the average reward per step against the exploration parameter, one panel per policy.

    Args:
        curves (dict): mapping from policy name to the pair of the parameter values and the average reward per
            step at each of them;
        symbols (dict): mapping from policy name to the symbol of its exploration parameter, used as x label;
        title (str): the title of the figure, naming the quantity plotted.

    Returns:
        The figure holding the panels.

    """
    fig, axes = plt.subplots(1, len(curves), figsize=(4 * len(curves), 3.5), sharey=True)
    fig.suptitle(title)

    for ax, (name, (values, rewards)) in zip(axes, curves.items()):
        ax.plot(values, rewards)
        ax.set_title(name)
        ax.set_xlabel(symbols[name])

    axes[0].set_ylabel('reward per step')
    fig.tight_layout()

    return fig


def experiment(policy_class, value, n_steps, n_steps_eval, seed):
    np.random.seed(seed)

    # MDP
    mdp = Taxi.generate()

    # Policy
    pi = policy_class(Parameter(value=value))

    # Agent
    learning_rate = Parameter(value=.3)
    agent = SARSA(mdp.info, pi, learning_rate=learning_rate)

    # Algorithm
    collect_dataset = CollectDataset()
    core = Core(agent, mdp, callbacks_fit=[collect_dataset])

    # Train
    core.learn(n_steps=n_steps, n_steps_per_fit=1, quiet=True)
    mean_reward_online = collect_dataset.get().reward.sum() / n_steps

    # Evaluate the greedy policy of the learned action values
    dataset = core.evaluate(n_steps=n_steps_eval, quiet=True, greedy=True)
    mean_reward_greedy = dataset.reward.sum() / n_steps_eval

    return mean_reward_online, mean_reward_greedy


if __name__ == '__main__':
    n_experiment = 25
    n_steps = 300000
    n_steps_eval = 20000
    policies = [EpsGreedy, Boltzmann, Mellowmax]
    ranges = {EpsGreedy: np.linspace(.05, .5, 10),
              Boltzmann: np.linspace(.5, 10, 10),
              Mellowmax: np.linspace(.5, 10, 10)}
    symbols = {EpsGreedy.name(): r'$\epsilon$',
               Boltzmann.name(): r'$\beta$',
               Mellowmax.name(): r'$\omega$'}

    logger = Logger('taxi_mellowmax', results_dir=get_log_dir(__file__))
    logger.log_experiment_info(SARSA, n_experiment=n_experiment, n_steps=n_steps, n_steps_eval=n_steps_eval)

    online_curves = dict()
    greedy_curves = dict()
    for policy_class in policies:
        logger.info(f'Policy: {policy_class.name()}')

        Js = list()
        for value in ranges[policy_class]:
            out = np.array(Parallel(n_jobs=-1)(delayed(experiment)(policy_class, value, n_steps, n_steps_eval, seed)
                                               for seed in range(n_experiment)))
            Js.append(out.mean(0))
        Js = np.array(Js)

        name = policy_class.name()
        logger.log_numpy_array(**{name + '_values': ranges[policy_class], name + '_online': Js[:, 0],
                                  name + '_greedy': Js[:, 1]})

        online_curves[name] = (ranges[policy_class], Js[:, 0])
        greedy_curves[name] = (ranges[policy_class], Js[:, 1])

    fig = plot_results(online_curves, symbols, 'Reward per step of the behaviour policy while learning')
    fig.savefig(logger.path / 'taxi_online.png')
    fig = plot_results(greedy_curves, symbols, 'Reward per step of the greedy policy after learning')
    fig.savefig(logger.path / 'taxi_greedy.png')
