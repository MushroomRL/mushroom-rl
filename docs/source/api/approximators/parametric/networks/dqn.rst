DQN networks
============

The architectures the DQN variants require: they change what the network outputs — an advantage and a value stream,
a distribution over returns, a set of quantiles — and therefore come paired with the algorithm that consumes them.

.. automodule:: mushroom_rl.approximators.parametric.networks.dueling_network

.. automodule:: mushroom_rl.approximators.parametric.networks.noisy_network

.. automodule:: mushroom_rl.approximators.parametric.networks.categorical_network

.. automodule:: mushroom_rl.approximators.parametric.networks.quantile_network

.. automodule:: mushroom_rl.approximators.parametric.networks.rainbow_network
