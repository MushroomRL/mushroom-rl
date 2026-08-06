Environments
============

All environments implement the ``Environment`` interface. Some of them, e.g. grid worlds, are finite Markov Decision
Processes: they extend the ``FiniteMDP`` class, exposing the transition probability matrix, the reward matrix and the
initial state distribution, so that they can also be solved with the dynamic programming solvers.

An environment is built either by calling its constructor directly, or by name through
:meth:`~mushroom_rl.core.Environment.make`, which looks it up in the registry every environment adds itself to with
``register``. Environments standing for a family of tasks accept the task after a ``'.'`` separator, e.g.
``Environment.make('Gymnasium.Pendulum-v1')``.

The tables below index every registered environment. The names are the keys :meth:`~mushroom_rl.core.Environment.make`
accepts. The spaces are the ones the environment reports in ``MDPInfo`` when built with its default arguments; where the
environment stands for a family of tasks they are task dependent, and where it is built from a user-supplied model
— ``FiniteMDP`` and ``LQR`` —  they follow the size of that model. Environments requiring an extra dependency are
registered only if that dependency imports successfully, so an environment missing from
``Environment.list_registered()`` means its extra is not installed.

.. toctree::
   :maxdepth: 1

   classic/index
   finite/index
   external/index
   mujoco/index
   pybullet/index
   isaacsim

.. rubric:: Built-in

.. csv-table::
   :header: "``make()`` name", "Class", "Observations", "Actions", "Extra dependency"
   :widths: 22, 24, 18, 16, 20

   "CarOnHill", ":class:`~mushroom_rl.environments.car_on_hill.CarOnHill`", "Box (2)", "Discrete (2)", "—"
   "CartPole", ":class:`~mushroom_rl.environments.cart_pole.CartPole`", "Box (2)", "Discrete (3)", "—"
   "FiniteMDP", ":class:`~mushroom_rl.environments.finite_mdp.FiniteMDP`", "Discrete", "Discrete", "—"
   "GridWorld", ":class:`~mushroom_rl.environments.grid_world.GridWorld`", "Discrete (9)", "Discrete (4)", "—"
   "GridWorldVanHasselt", ":class:`~mushroom_rl.environments.grid_world_van_hasselt.GridWorldVanHasselt`", "Discrete (10)", "Discrete (4)", "—"
   "InvertedPendulum", ":class:`~mushroom_rl.environments.inverted_pendulum.InvertedPendulum`", "Box (2)", "Box (1)", "—"
   "LQR", ":class:`~mushroom_rl.environments.lqr.LQR`", "Box", "Box", "—"
   "PuddleWorld", ":class:`~mushroom_rl.environments.puddle_world.PuddleWorld`", "Box (2)", "Discrete (5)", "—"
   "Segway", ":class:`~mushroom_rl.environments.segway.Segway`", "Box (3)", "Box (1)", "—"
   "ShipSteering", ":class:`~mushroom_rl.environments.ship_steering.ShipSteering`", "Box (4)", "Box (1)", "—"
   "SimpleChain", ":class:`~mushroom_rl.environments.simple_chain.SimpleChain`", "Discrete (5)", "Discrete (2)", "—"
   "Taxi", ":class:`~mushroom_rl.environments.taxi.Taxi`", "Discrete (252)", "Discrete (4)", "—"

.. rubric:: External libraries

.. csv-table::
   :header: "``make()`` name", "Class", "Observations", "Actions", "Extra dependency"
   :widths: 22, 24, 18, 16, 20

   "Atari", ":class:`~mushroom_rl.environments.atari.Atari`", "Box (image)", "Discrete", "``mushroom_rl[atari]``"
   "Gymnasium", ":class:`~mushroom_rl.environments.gymnasium_env.Gymnasium`", "task dependent", "task dependent", "``mushroom_rl[gymnasium]``"
   "DMControl", ":class:`~mushroom_rl.environments.dm_control_env.DMControl`", "task dependent", "task dependent", "``mushroom_rl[dm_control]``"
   "MiniGrid", ":class:`~mushroom_rl.environments.minigrid_env.MiniGrid`", "Box (3, H, W)", "Discrete", "``mushroom_rl[minigrid]``"
   "MiniGridRGB", ":class:`~mushroom_rl.environments.minigrid_env.MiniGridRGB`", "Box (grayscale image)", "Discrete", "``mushroom_rl[minigrid]``"

.. rubric:: MuJoCo

.. csv-table::
   :header: "``make()`` name", "Class", "Observations", "Actions", "Extra dependency"
   :widths: 22, 24, 18, 16, 20

   "Ant", ":class:`~mushroom_rl.environments.mujoco_envs.ant.Ant`", "Box (27)", "Box (8)", "``mushroom_rl[mujoco]``"
   "HalfCheetah", ":class:`~mushroom_rl.environments.mujoco_envs.half_cheetah.HalfCheetah`", "Box (17)", "Box (6)", "``mushroom_rl[mujoco]``"
   "Hopper", ":class:`~mushroom_rl.environments.mujoco_envs.hopper.Hopper`", "Box (11)", "Box (3)", "``mushroom_rl[mujoco]``"
   "Walker2D", ":class:`~mushroom_rl.environments.mujoco_envs.walker_2d.Walker2D`", "Box (17)", "Box (6)", "``mushroom_rl[mujoco]``"
   "BallInACup", ":class:`~mushroom_rl.environments.mujoco_envs.ball_in_a_cup.BallInACup`", "Box (23)", "Box (7)", "``mushroom_rl[mujoco]``"
   "Reach", ":class:`~mushroom_rl.environments.mujoco_envs.reach.Reach`", "Box (32)", "Box (7)", "``mushroom_rl[mujoco]``"
   "Push", ":class:`~mushroom_rl.environments.mujoco_envs.push.Push`", "Box (37)", "Box (7)", "``mushroom_rl[mujoco]``"
   "Pick", ":class:`~mushroom_rl.environments.mujoco_envs.pick.Pick`", "Box (45)", "Box (8)", "``mushroom_rl[mujoco]``"
   "PegInsertion", ":class:`~mushroom_rl.environments.mujoco_envs.peg_insertion.PegInsertion`", "Box (45)", "Box (7)", "``mushroom_rl[mujoco]``"
   "AirHockeyHit", ":class:`~mushroom_rl.environments.mujoco_envs.air_hockey.hit.AirHockeyHit`", "Box (13)", "Box (3)", "``mushroom_rl[mujoco]``"
   "AirHockeyDefend", ":class:`~mushroom_rl.environments.mujoco_envs.air_hockey.defend.AirHockeyDefend`", "Box (13)", "Box (3)", "``mushroom_rl[mujoco]``"
   "AirHockeyPrepare", ":class:`~mushroom_rl.environments.mujoco_envs.air_hockey.prepare.AirHockeyPrepare`", "Box (13)", "Box (3)", "``mushroom_rl[mujoco]``"
   "AirHockeyRepel", ":class:`~mushroom_rl.environments.mujoco_envs.air_hockey.repel.AirHockeyRepel`", "Box (13)", "Box (3)", "``mushroom_rl[mujoco]``"

.. rubric:: PyBullet

.. csv-table::
   :header: "``make()`` name", "Class", "Observations", "Actions", "Extra dependency"
   :widths: 22, 24, 18, 16, 20

   "AirHockeyHitBullet", ":class:`~mushroom_rl.environments.pybullet_envs.air_hockey.hit.AirHockeyHitBullet`", "Box (12)", "Box (3)", "``mushroom_rl[bullet]``"
   "AirHockeyDefendBullet", ":class:`~mushroom_rl.environments.pybullet_envs.air_hockey.defend.AirHockeyDefendBullet`", "Box (12)", "Box (3)", "``mushroom_rl[bullet]``"
   "AirHockeyPrepareBullet", ":class:`~mushroom_rl.environments.pybullet_envs.air_hockey.prepare.AirHockeyPrepareBullet`", "Box (12)", "Box (3)", "``mushroom_rl[bullet]``"
   "AirHockeyRepelBullet", ":class:`~mushroom_rl.environments.pybullet_envs.air_hockey.repel.AirHockeyRepelBullet`", "Box (12)", "Box (3)", "``mushroom_rl[bullet]``"

.. rubric:: Isaac Sim

The Isaac Sim environments are **not** registered: ``mushroom_rl/environments/__init__.py`` does not import
``isaacsim_envs``, so they are not reachable through :meth:`~mushroom_rl.core.Environment.make` and must be
imported from their module explicitly, e.g.
``from mushroom_rl.environments.isaacsim_envs.cartpole import CartPoleIsaac``. They are vectorized environments:
they
step a batch of simulated copies at once, and are driven through the vectorized ``Core``.
