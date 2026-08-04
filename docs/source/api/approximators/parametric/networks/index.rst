Networks
========

Pre-built PyTorch network architectures for use with ``TorchApproximator``. They are ordinary ``torch.nn.Module``
subclasses: they are given to the approximator through its ``network`` argument, and the shapes are passed by the
approximator itself.

.. autosummary::
   :nosignatures:

   ~mushroom_rl.approximators.parametric.networks.linear_network.LinearNetwork
   ~mushroom_rl.approximators.parametric.networks.actor_network.ActorNetwork
   ~mushroom_rl.approximators.parametric.networks.critic_network.CriticNetwork
   ~mushroom_rl.approximators.parametric.networks.q_network.QNetwork
   ~mushroom_rl.approximators.parametric.networks.atari_network.AtariNetwork
   ~mushroom_rl.approximators.parametric.networks.recurrent_network.RecurrentActorNetwork
   ~mushroom_rl.approximators.parametric.networks.recurrent_network.RecurrentCriticNetwork
   ~mushroom_rl.approximators.parametric.networks.dueling_network.DuelingNetwork
   ~mushroom_rl.approximators.parametric.networks.noisy_network.NoisyNetwork
   ~mushroom_rl.approximators.parametric.networks.categorical_network.CategoricalNetwork
   ~mushroom_rl.approximators.parametric.networks.quantile_network.QuantileNetwork
   ~mushroom_rl.approximators.parametric.networks.rainbow_network.RainbowNetwork

.. toctree::
   :maxdepth: 1

   basic
   atari
   recurrent
   dqn
