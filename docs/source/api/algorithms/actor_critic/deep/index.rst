Deep actor-critic
=================

.. module:: mushroom_rl.algorithms.actor_critic.deep_actor_critic

The deep actor-critic algorithms represent both actor and critic with PyTorch networks. They split in two groups,
which is also how their base classes are organized:

- **off-policy** methods, :class:`DDPG`, :class:`TD3` and :class:`SAC`, learn from a replay memory and keep target
  networks. They update the actor by backpropagating the critic's Q-value through the action the actor produces: the
  deterministic action for DDPG and TD3, a reparametrized sample for SAC. They build the policy themselves, so they
  act on continuous actions;
- **on-policy** methods fit on the batch of samples they just collected, with a state-value critic estimating the
  advantage of the actions taken, and take the policy from the caller, so the action space is the one their policy
  supports. :class:`A2C` estimates the advantage from Monte Carlo returns; :class:`TRPO` and :class:`PPO` use
  generalized advantage estimation, and keep every policy update small, TRPO within a trust region and PPO by
  clipping the probability ratio. :class:`PPO_BPTT` extends PPO to recurrent policies, and :class:`RudinPPO` adds a
  learning rate adapted to the KL divergence.

Every algorithm but TRPO takes its actor optimizer as a dictionary of the optimizer class and its parameters. Gradient
clipping is added with a ``'clipping'`` entry, holding the clipping function and its parameters:

.. code-block:: python

    actor_optimizer = {'class': optim.Adam, 'params': {'lr': 1e-3},
                       'clipping': {'method': torch.nn.utils.clip_grad_norm_, 'params': {'max_norm': 1.}}}

Every one of them can also be built as an *asymmetric* actor-critic, whose actor and critic read different entries
of the observation, each with its own history length, e.g. a critic reading privileged information available in
simulation only. :meth:`~mushroom_rl.algorithms.actor_critic.deep_actor_critic.DeepAC.asymmetric` builds it from the
constructor parameters of the algorithm, the actor entries, the critic entries and the critic history length:

.. code-block:: python

    agent = RudinPPO.asymmetric(mdp.info, policy, critic_params=critic_params, actor_indices=actor_indices,
                                critic_history_length=1, history_length=8, **ppo_params)

The base classes these algorithms extend, to build a new one on, are listed in the following table and documented in
:doc:`deep_actor_critic`.

.. autosummary::
   :nosignatures:

   DeepAC
   OnPolicyDeepAC
   OffPolicyDeepAC

.. toctree::
   :maxdepth: 1

   deep_actor_critic
   off_policy
   on_policy
