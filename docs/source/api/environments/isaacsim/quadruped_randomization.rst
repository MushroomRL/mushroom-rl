Quadruped domain randomization
==============================

``QuadrupedRandomizer`` is not an environment: it is the description of how far every physical and control parameter
may stray from its nominal value, and of which value each parallel environment currently runs with.
``QuadrupedIsaac`` owns one, seeds it with the nominal properties read out of the live simulation, writes back
whatever it draws, and reads the current parameters off it.

.. automodule:: mushroom_rl.environments.isaacsim_envs.quadruped_randomizer
    :private-members:
