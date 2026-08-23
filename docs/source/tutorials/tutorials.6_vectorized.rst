How to use the VectorizedEnvironment interface
==============================================

Some environments can step many copies of the same problem in parallel, which lets the
agent collect samples much faster. This is common when the simulator is natively batched,
for example GPU-based simulators that evolve a whole batch of states at once.

The vectorized interface
------------------------

Such environments extend the ``VectorizedEnvironment`` interface instead of
``Environment``. The constructor takes the usual ``MDPInfo`` together with the number of
parallel copies ``n_envs``, and instead of the single-environment ``reset``, ``step`` and
``render`` methods you implement their batched counterparts:

- ``reset_all(env_mask, state=None)``: reset the selected environments to their initial
  state, returning the batched initial states and a list of episode-info dictionaries;
- ``step_all(env_mask, action)``: apply a batch of actions to the selected environments,
  returning the batched next states, rewards, absorbing flags and a list of step-info
  dictionaries;
- ``render_all(env_mask, record=False)``: render the selected environments.

The recurring argument is the ``env_mask``: a boolean array of length ``n_envs`` that
selects which copies the operation applies to. This is what makes parallel collection
efficient — the copies run independent episodes that terminate at different times, so the
``Core`` only resets the ones that have just finished while the others keep stepping,
rather than restarting the whole batch in lockstep.

A ``VectorizedEnvironment`` is also a valid single environment: the base class implements
``reset``, ``step`` and ``render`` by forwarding to the batched methods on a single
*default* copy, which you can select with ``set_default_env``. This is mostly useful for
debugging or for rendering one copy of the batch.

Parallelizing a standard environment
------------------------------------

You do not need a natively batched simulator to benefit from this: any standard
environment can be parallelized across processes with ``MultiprocessEnvironment``, which
wraps several copies of it into a single ``VectorizedEnvironment``:

.. code-block:: python

    from mushroom_rl.core import MultiprocessEnvironment
    from mushroom_rl.environments import Gymnasium

    env = MultiprocessEnvironment(Gymnasium, 'Pendulum-v1', horizon=200, gamma=.99, n_envs=15)

``MultiprocessEnvironment`` takes the environment class followed by the arguments of its
constructor, plus the number of parallel copies ``n_envs``.

Each copy runs in its own process, forked from the main one, so it starts with a *copy* of
the generators of the parent. Reseeding them is what stops every copy from replaying the
very same trajectory. ``MultiprocessEnvironment.seed(seed)`` handles this by giving copy
``i`` the seed ``seed + i``.

.. warning::

    What is reseeded automatically in each worker is only the **global numpy and torch**
    random number generators. An environment that draws randomness from anywhere else — a
    generator of its own, or the internal RNG of a simulator it wraps — is *not* reseeded
    by this, and its copies will produce identical trajectories.

    Such an environment has to override :meth:`~mushroom_rl.core.Environment.seed` and
    reseed its own source there. ``MultiprocessEnvironment`` calls it, but only when the
    class actually overrides it: the base implementation does nothing and warns, so a
    missing override fails silently as far as the parallel copies are concerned.

Running the experiment
----------------------

You do not need to handle the batching yourself when running experiments: the ``Core``
recognizes a vectorized environment and runs the appropriate parallel collection loop
internally. Your experiment script is unchanged — you build and use the ``Core`` exactly
as before:

.. code-block:: python

    core = Core(agent, env)
    core.learn(n_steps=30000, n_steps_per_fit=3000)
