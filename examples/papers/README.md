# Paper reproductions

Scripts that reproduce experiments from a published paper. Hyperparameters, environments and evaluation
protocols are taken from the publication rather than chosen for convenience.

| Script                                                     | Reproduces                                                                                                        |
|------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| [`van_hasselt_double_q.py`](van_hasselt_double_q.py)       | *Double Q-Learning*, van Hasselt H., 2010 — grid world, comparing SARSA and several Q-Learning variants           |
| [`double_chain_q_learning.py`](double_chain_q_learning.py) | Q-Learning variants on the double chain of *Relative Entropy Policy Search*, Peters J. et al., 2010               |
| [`taxi_mellowmax.py`](taxi_mellowmax.py)                   | *An Alternative Softmax Operator for Reinforcement Learning*, Asadi K. et al., 2017 — taxi problem                |
| [`car_on_hill_fqi.py`](car_on_hill_fqi.py)                 | *Tree-Based Batch Mode Reinforcement Learning*, Ernst D. et al., 2005 — car on hill with FQI                      |
| [`cartpole_lspi.py`](cartpole_lspi.py)                     | *Least-Squares Policy Iteration*, Lagoudakis M. G. and Parr R., 2003 — inverted pendulum                          |
| [`mountain_car_sarsa.py`](mountain_car_sarsa.py)           | *True Online TD(lambda)*, van Seijen H. et al., 2014 — mountain car                                               |
| [`atari_dqn.py`](atari_dqn.py)                             | *Human-Level Control Through Deep Reinforcement Learning*, Mnih V. et al., 2015 — Atari with DQN and its variants |

## Cost

These are the most expensive examples in the repository. Several average over many independent runs
(`van_hasselt_double_q.py` and `taxi_mellowmax.py` use `joblib` to parallelise them), and `atari_dqn.py`
trains for tens of millions of frames at its published settings — use its `--debug` flag for a quick
smoke run. Reduce the number of runs or the step budget if you only want to see a script work.

Scripts that write results do so under `examples/logs/`.
