Classic control
===============

The classic control benchmarks, implemented in MushroomRL itself with no external simulator: their dynamics are
integrated directly, so they run anywhere and are cheap enough to be used in tests and tutorials. Each of them
reproduces the setup of the paper its docstring names.

.. autosummary::
   :nosignatures:

   ~mushroom_rl.environments.car_on_hill.CarOnHill
   ~mushroom_rl.environments.cart_pole.CartPole
   ~mushroom_rl.environments.inverted_pendulum.InvertedPendulum
   ~mushroom_rl.environments.lqr.LQR
   ~mushroom_rl.environments.puddle_world.PuddleWorld
   ~mushroom_rl.environments.segway.Segway
   ~mushroom_rl.environments.ship_steering.ShipSteering

.. toctree::
   :maxdepth: 1

   car_on_hill
   cart_pole
   inverted_pendulum
   lqr
   puddle_world
   segway
   ship_steering
