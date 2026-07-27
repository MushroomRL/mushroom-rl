# Algorithms

One script per algorithm, on an environment chosen to be cheap enough that the run finishes quickly. These
are the templates to copy when starting your own experiment: each shows how to build the approximators, the
policy and the agent, and how to drive `Core` through a learning loop.

Split by the three families of `mushroom_rl.algorithms`.

## [`value/`](value) — value-based methods

| Script | Algorithm | Environment |
|---|---|---|
| [`simple_chain_qlearning.py`](value/simple_chain_qlearning.py) | Q-Learning | simple chain |
| [`grid_world_sarsa.py`](value/grid_world_sarsa.py) | SARSA(λ) | grid world or taxi, selected with `--env` |
| [`puddle_world_sarsa.py`](value/puddle_world_sarsa.py) | True Online SARSA(λ) with tile coding | puddle world |
| [`acrobot_dqn.py`](value/acrobot_dqn.py) | DQN | Acrobot |
| [`minigrid_dqn.py`](value/minigrid_dqn.py) | DQN and its variants, selected on the command line | MiniGrid, from pixels |

`grid_world_sarsa.py` also computes the optimal return exactly with value iteration, and reports it next to
the learned one as a reference.

## [`actor_critic/`](actor_critic) — actor-critic methods

| Script | Algorithm | Environment |
|---|---|---|
| [`pendulum_ac.py`](actor_critic/pendulum_ac.py) | Stochastic actor-critic with average reward | inverted pendulum |
| [`pendulum_dpg.py`](actor_critic/pendulum_dpg.py) | COPDAC-Q, deterministic policy gradient | inverted pendulum |
| [`pendulum_a2c.py`](actor_critic/pendulum_a2c.py) | A2C | Pendulum |
| [`acrobot_a2c.py`](actor_critic/acrobot_a2c.py) | A2C | Acrobot |
| [`pendulum_ddpg.py`](actor_critic/pendulum_ddpg.py) | DDPG and TD3 | Pendulum |
| [`pendulum_sac.py`](actor_critic/pendulum_sac.py) | SAC | Pendulum |
| [`pendulum_trust_region.py`](actor_critic/pendulum_trust_region.py) | PPO and TRPO | Pendulum |
| [`gym_recurrent_ppo.py`](actor_critic/gym_recurrent_ppo.py) | PPO with a recurrent policy trained through BPTT | HalfCheetah |

The first two are the classic, non-deep actor-critics; the rest are the deep ones.

## [`policy_search/`](policy_search) — episode-based policy search

| Script | Algorithms | Environment |
|---|---|---|
| [`lqr_pg.py`](policy_search/lqr_pg.py) | REINFORCE, GPOMDP, eNAC | LQR |
| [`lqr_bbo.py`](policy_search/lqr_bbo.py) | REPS, RWR, PGPE, ConstrainedREPS, MORE | LQR |
| [`segway_bbo.py`](policy_search/segway_bbo.py) | REPS, RWR, PGPE | segway |
| [`segway_eppo.py`](policy_search/segway_eppo.py) | ePPO | segway |
| [`ship_steering_bbo.py`](policy_search/ship_steering_bbo.py) | REPS, RWR, PGPE with tile coding | ship steering |

`lqr_pg.py` optimises the policy parameters directly, while the others search in the parameter space of a
distribution over policies.
