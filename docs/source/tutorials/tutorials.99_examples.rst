Usage Examples
==============

In the following, we collect the links to MushroomRL scripts showing examples for most approaches available in
MushroomRL.

The examples can be all found in the `examples <https://github.com/MushroomRL/mushroom-rl/tree/dev/examples>`_ folder
in the MushroomRL `repository <https://github.com/MushroomRL/mushroom-rl>`_. They are grouped in four folders:
`papers <https://github.com/MushroomRL/mushroom-rl/tree/dev/examples/papers>`_ reproduces experiments from a
published paper, `algorithms <https://github.com/MushroomRL/mushroom-rl/tree/dev/examples/algorithms>`_ collects one
script per algorithm, `environments <https://github.com/MushroomRL/mushroom-rl/tree/dev/examples/environments>`_ shows
the interface of a specific environment class or simulator, and
`tools <https://github.com/MushroomRL/mushroom-rl/tree/dev/examples/tools>`_ demonstrates the functionalities the
library provides. Each of those folders carries its own README, describing its contents in more detail.

Every script can be launched directly, from any working directory:

.. code-block:: bash

    python examples/algorithms/value/simple_chain_qlearning.py


Paper Reproductions
-------------------

- `Grid World of Van Hasselt <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/papers/van_hasselt_double_q.py>`_ — *Double Q-Learning*, van Hasselt H., 2010
- `Double Chain <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/papers/double_chain_q_learning.py>`_ — Q-Learning variants on the double chain of *Relative Entropy Policy Search*, Peters J. et al., 2010
- `Taxi with Mellowmax <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/papers/taxi_mellowmax.py>`_ — *An Alternative Softmax Operator for Reinforcement Learning*, Asadi K. et al., 2017
- `CarOnHill with FQI <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/papers/car_on_hill_fqi.py>`_ — *Tree-Based Batch Mode Reinforcement Learning*, Ernst D. et al., 2005
- `CartPole with LSPI <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/papers/cartpole_lspi.py>`_ — *Least-Squares Policy Iteration*, Lagoudakis M. G. and Parr R., 2003
- `Atari with DQN <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/papers/atari_dqn.py>`_ — *Human-Level Control Through Deep Reinforcement Learning*, Mnih V. et al., 2015

These are the most expensive examples of the repository: several of them average over many independent runs, and
the Atari one trains for tens of millions of frames at its published settings.


Value-Based Algorithms
----------------------

- `Simple Chain with Q-Learning <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/algorithms/value/simple_chain_qlearning.py>`_
- `Grid World and Taxi with SARSA(lambda) <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/algorithms/value/grid_world_sarsa.py>`_
- `Puddle World with True Online SARSA(lambda) <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/algorithms/value/puddle_world_sarsa.py>`_
- `Mountain Car with True Online SARSA(lambda) <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/algorithms/value/mountain_car_sarsa.py>`_
- `Acrobot with DQN <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/algorithms/value/acrobot_dqn.py>`_
- `MiniGrid with DQN and its variants <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/algorithms/value/minigrid_dqn.py>`_


Classical Policy Search and Actor-Critic
----------------------------------------

- `LQR with Policy Gradient <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/algorithms/policy_search/lqr_pg.py>`_
- `Pendulum with Stochastic Actor-Critic <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/algorithms/actor_critic/pendulum_ac.py>`_
- `Pendulum with Deterministic Actor-Critic <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/algorithms/actor_critic/pendulum_dpg.py>`_


Black Box Optimization
----------------------

- `LQR with BBO <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/algorithms/policy_search/lqr_bbo.py>`_
- `Segway with BBO <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/algorithms/policy_search/segway_bbo.py>`_
- `Segway with ePPO <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/algorithms/policy_search/segway_eppo.py>`_
- `Ship Steering with BBO <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/algorithms/policy_search/ship_steering_bbo.py>`_


Deep Actor-Critic
-----------------

- `Pendulum with A2C <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/algorithms/actor_critic/pendulum_a2c.py>`_
- `Acrobot with A2C <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/algorithms/actor_critic/acrobot_a2c.py>`_
- `Pendulum with Trust Region approaches <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/algorithms/actor_critic/pendulum_trust_region.py>`_
- `Pendulum with Deterministic Gradient <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/algorithms/actor_critic/pendulum_ddpg.py>`_
- `Pendulum with SAC <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/algorithms/actor_critic/pendulum_sac.py>`_
- `HalfCheetah with Recurrent PPO <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/algorithms/actor_critic/gym_recurrent_ppo.py>`_


MuJoCo Environments
-------------------

- `Locomotion tasks with PPO <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/environments/mujoco/locomotion_ppo.py>`_
- `Manipulation tasks with PPO <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/environments/mujoco/manipulation_ppo.py>`_
- `Air Hockey with SAC <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/environments/mujoco/air_hockey_sac.py>`_


Continuous Control From Pixels
------------------------------

- `Walker (Stand Task) from Pixel <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/environments/dm_control/walker_stand_ddpg.py>`_
- `Walker (Stand Task) from Pixel and Shared Network <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/environments/dm_control/walker_stand_ddpg_shared_net.py>`_


Vectorized Environments
-----------------------

- `Parallel Pendulums with Trust Region approaches <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/environments/multiprocess_environment/pendulum_trust_region.py>`_
- `Parallel Segways with BBO <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/environments/multiprocess_environment/segway_bbo.py>`_
- `IsaacSim CartPole with PPO <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/environments/isaacsim/cartpole_ppo.py>`_
- `IsaacSim Unitree A1 with PPO <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/environments/isaacsim/a1_rudin_ppo.py>`_
- `IsaacSim Honey Badger with PPO <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/environments/isaacsim/honey_badger_ppo.py>`_
- `IsaacSim Silver Badger with PPO <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/environments/isaacsim/silver_badger_ppo.py>`_

The two ``multiprocess_environment`` scripts have a single-environment twin under ``algorithms/``, so that diffing a
script against its twin shows what vectorization costs.


Others Examples (Tools)
-----------------------

- `Logging, Weights & Biases and Video Recording <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/tools/wandb_logging.py>`_
- `Video Recording of a Vectorized Environment <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/tools/multiprocess_env_recording.py>`_
- `Using Dataset Monitoring callback and State Normalization <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/tools/monitoring_and_normalization.py>`_
- `Using the list Dataset backend <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/tools/list_backend.py>`_
- `Using the Finite MDP Viewer <https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/tools/gridworld_viewer.py>`_
