# Algorithms

One script per algorithm, on an environment chosen to be cheap enough that the run finishes quickly. These are the
templates to copy when starting your own experiment: each shows how to build the approximators, the policy and the
agent, and how to drive `Core` through a learning loop.

Split by the three families of `mushroom_rl.algorithms`.

## [`value/`](value) — value-based methods

| Script                                                         | Algorithm                                          | Environment                              |
|----------------------------------------------------------------|----------------------------------------------------|------------------------------------------|
| [`simple_chain_qlearning.py`](value/simple_chain_qlearning.py) | Q-Learning                                         | SimpleChain                              |
| [`grid_world_sarsa.py`](value/grid_world_sarsa.py)             | SARSA(λ)                                           | GridWorld or Taxi, selected with `--env` |
| [`puddle_world_sarsa.py`](value/puddle_world_sarsa.py)         | True Online SARSA(λ) with tile coding              | PuddleWorld                              |
| [`mountain_car_sarsa.py`](value/mountain_car_sarsa.py)         | True Online SARSA(λ) with tile coding              | MountainCar                              |
| [`acrobot_dqn.py`](value/acrobot_dqn.py)                       | DQN                                                | Acrobot                                  |
| [`minigrid_dqn.py`](value/minigrid_dqn.py)                     | DQN and its variants, selected on the command line | MiniGrid, from pixels                    |

`grid_world_sarsa.py` also computes the optimal return exactly with value iteration, and reports it next to the learned
one as a reference.

## [`actor_critic/`](actor_critic) — actor-critic methods

| Script                                                                      | Algorithm                                        | Environment      |
|-----------------------------------------------------------------------------|--------------------------------------------------|------------------|
| [`inverted_pendulum_ac.py`](actor_critic/inverted_pendulum_ac.py)           | Stochastic actor-critic with average reward      | InvertedPendulum |
| [`inverted_pendulum_dpg.py`](actor_critic/inverted_pendulum_dpg.py)         | COPDAC-Q, deterministic policy gradient          | InvertedPendulum |
| [`gym_pendulum_a2c.py`](actor_critic/gym_pendulum_a2c.py)                   | A2C                                              | Pendulum         |
| [`gym_acrobot_a2c.py`](actor_critic/gym_acrobot_a2c.py)                     | A2C                                              | Acrobot          |
| [`gym_pendulum_ddpg.py`](actor_critic/gym_pendulum_ddpg.py)                 | DDPG or TD3, selected with `--alg`               | Pendulum         |
| [`gym_pendulum_sac.py`](actor_critic/gym_pendulum_sac.py)                   | SAC                                              | Pendulum         |
| [`gym_pendulum_trust_region.py`](actor_critic/gym_pendulum_trust_region.py) | PPO or TRPO, selected with `--alg`               | Pendulum         |
| [`gym_recurrent_ppo.py`](actor_critic/gym_recurrent_ppo.py)                 | PPO with a recurrent policy trained through BPTT | HalfCheetah      |

The first two are the classic actor-critics approaches; the rest are the deep actor-critic algorithms.

Every script here ends with a visualization of the learned policy, which `--no-render` skips for an unattended run.
`gym_pendulum_sac.py` additionally takes `--save` to store the best agent of the run, and `--load` to start from the best
agent of an earlier one, given the log directory that run wrote into.

## [`policy_search/`](policy_search) — policy search

| Script                                                       | Algorithms                             | Environment  |
|--------------------------------------------------------------|----------------------------------------|--------------|
| [`lqr_pg.py`](policy_search/lqr_pg.py)                       | REINFORCE, GPOMDP, eNAC                | LQR          |
| [`lqr_bbo.py`](policy_search/lqr_bbo.py)                     | REPS, RWR, PGPE, ConstrainedREPS, MORE | LQR          |
| [`segway_bbo.py`](policy_search/segway_bbo.py)               | REPS, RWR, PGPE                        | Segway       |
| [`segway_eppo.py`](policy_search/segway_eppo.py)             | ePPO                                   | Segway       |
| [`ship_steering_bbo.py`](policy_search/ship_steering_bbo.py) | REPS, RWR, PGPE with tile coding       | ShipSteering |

`lqr_pg.py` is a traditional policy gradient method in action space, while the others search in the parameter space of a
distribution over policies.
