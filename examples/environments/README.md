# Environments

Examples centred on an environment class rather than on an algorithm: how to construct it, what its observations and
actions look like, and a reasonable algorithm to solve it. Each subfolder contains examples per environment class or
simulator, each with its own optional dependency.

## [`mujoco/`](mujoco)

Install with `pip install mushroom_rl[mujoco]`.

| Script                                              | Environments                                                                  |
|-----------------------------------------------------|-------------------------------------------------------------------------------|
| [`locomotion_ppo.py`](mujoco/locomotion_ppo.py)     | Ant, HalfCheetah, Hopper and Walker2D, selected with `--env`, solved with PPO |
| [`manipulation_ppo.py`](mujoco/manipulation_ppo.py) | Reach, Push, Pick and PegInsertion, selected with `--env`, solved with PPO    |
| [`air_hockey_sac.py`](mujoco/air_hockey_sac.py)     | the air hockey hitting task, with SAC                                         |

## [`dm_control/`](dm_control)

The DeepMind Control Suite, also installed with `pip install mushroom_rl[mujoco]`.

| Script                                                                          | Shows                                                                                                                                               |
|---------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| [`walker_stand_ddpg.py`](dm_control/walker_stand_ddpg.py)                       | the walker stand task **from pixels**, with DDPG; actor and critic each carry their own convolutional encoder                                       |
| [`walker_stand_ddpg_shared_net.py`](dm_control/walker_stand_ddpg_shared_net.py) | the same task, with the convolutional encoder factored into a separate embedding module shared by actor and critic, which then reduce to plain MLPs |

Together they are the reference for learning from images, and show the two ways of arranging the encoder.

# Vectorized environments

The subfolders below hold `VectorizedEnvironment` classes, which step many copies of an environment at once and expose
batched observations, rewards and terminations. `Core` detects them and dispatches to its vectorized implementation, so
the agent code is the same as in a single-environment experiment.

## [`multiprocess_environment/`](multiprocess_environment)

`MultiprocessEnvironment` wraps any ordinary MushroomRL environment, running copies of it in separate worker processes
and gathering their outputs into batches. It works with every environment in the library and needs no extra dependency.

| Script                                                                                  | Environment                                              |
|-----------------------------------------------------------------------------------------|----------------------------------------------------------|
| [`gym_pendulum_trust_region.py`](multiprocess_environment/gym_pendulum_trust_region.py) | 15 parallel Pendulums, PPO or TRPO selected with `--alg` |
| [`segway_bbo.py`](multiprocess_environment/segway_bbo.py)                               | 15 parallel segways, black-box policy search             |

Both have a single-environment twin under [`../algorithms/`](../algorithms) — `gym_pendulum_trust_region.py`
and `segway_bbo.py`, in `actor_critic/` and `policy_search/` respectively. The pairs are deliberately kept close, so
diffing a script against its twin shows what vectorization costs: for the segway that is the environment construction
and nothing else.

## [`isaacsim/`](isaacsim)

`IsaacSim` is natively batched: a single GPU simulation advances thousands of environments at once, rather than running
many processes. Used here for legged locomotion and a cartpole.

| Script                                                                | Environment                                                      |
|-----------------------------------------------------------------------|------------------------------------------------------------------|
| [`cartpole_ppo.py`](isaacsim/cartpole_ppo.py)                         | 64 cartpoles, PPO                                                |
| [`quadruped_locomotion_ppo.py`](isaacsim/quadruped_locomotion_ppo.py) | 4096 quadrupeds, PPO — `--robot {a1,honey_badger,silver_badger}` |

These need NVIDIA Isaac Sim, which is a large install and requires a capable GPU. Follow the
[Isaac Sim installation guide](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/index.html).
