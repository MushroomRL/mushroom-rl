How to stack a history of observations
======================================

In many problems a single observation does not give enough information about the state of the environment, and
the agent must rely on a short window of the recent past to take the optimal action, e.g. a stack of the last
few frames to perceive velocities, or the last few actions it took. In MushroomRL this windowing is handled by
the :class:`~mushroom_rl.core.history_manager.HistoryManager`, an object that assembles the per-timestep
**context** fed to the policy, i.e. the stacked window of the most recent entries of one or more streams.

The context is kept deliberately separate from the *latent* policy state (see the
:doc:`../mushroom_rl.policy` documentation). The distinction is what a quantity depends on:

- the **latent policy state** cannot be recomputed from the observed trajectory, e.g. a recurrent hidden state or
  the Ornstein-Uhlenbeck noise. It must therefore be stored in the dataset as policy state;
- the **context** is a *deterministic function of the observed trajectory*, e.g. a window of stacked
  observations. It can always be rebuilt from the stored transitions, so it is assembled on the fly by the
  history manager and never stored.

By design, the manager is the single entry point for building the context both online and offline: the same
object that stacks the streams while the agent interacts with the environment also rebuilds them from a replay
memory, guaranteeing that every stream is reconstructed with the same rule and that what the policy sees during
learning matches exactly what it saw while acting.

The multi-stream design
-----------------------

A history manager holds an ordered set of named **streams**. Each stream is described by a few parameters:

- ``length`` — the number of consecutive entries stacked into the window;
- ``offset`` — how many steps behind the current one the window ends (0 means it ends at the current step);
- ``shape`` and ``dtype`` — the shape and data type of a single entry, so a window is a ``(length, *shape)``
  array, squeezed back to ``(*shape)`` when ``length`` is 1.

By default, two streams are handled automatically:

- ``obs_history`` — the reserved observation stream; stacks the last ``history_length`` observations with ``offset`` 0
  (its window ends at the current step), and is passed to the policy in place of the ``state`` argument;
- ``action_history`` — stacks the last ``action_history_length`` actions with ``offset`` 1 (its window ends one
  step behind, since the current action has not been taken yet), and is passed to the policy as a keyword argument.

The two are orthogonal: either can be active on its own, or both together, and each keeps its own length. The
usual way to build a manager is :meth:`~mushroom_rl.core.history_manager.HistoryManager.from_infos`, which reads
the shapes and data types from the MDP and agent information and returns ``None`` when no stream is active
(``history_length`` 1 and ``action_history_length`` 0). In the following we construct a manager directly and
exercise it, to illustrate its behaviour:

.. literalinclude:: code/history_manager.py
   :lines: 1-17

The :attr:`~mushroom_rl.core.history_manager.HistoryManager.max_reach` property reports the deepest backward
reach across all streams (the maximum of ``offset + length - 1``); it is how a circular replay buffer knows how
many of its oldest samples to keep clear of the write head so every window can be rebuilt.

In practice you rarely build the manager by hand. Algorithms that support a history expose ``history_length``
and/or ``action_history_length`` as constructor parameters and forward them to the
:class:`~mushroom_rl.core.Agent`, which wires up the manager for you and injects it wherever it is needed (the
policy call and the replay memory). The rest of this tutorial uses the manager directly only to show what happens
under the hood.

Stacking online
---------------

While the agent interacts with the environment, the manager stacks the streams one step at a time. At each step
it is called with the current observation, appends the entries to its internal buffers, and returns a tuple
``(state, policy_kwargs)`` ready to be used as ``policy.draw_action(state, **policy_kwargs)``: the observation
window takes the place of ``state``, and every other stream is returned under its own name. The ``action_history``
stream is not passed in; the action drawn at each step is instead reported back through
:meth:`~mushroom_rl.core.history_manager.HistoryManager.record_action`, and the manager uses it as the most recent
entry of the window at the next step (realizing its ``offset`` 1). A stream whose value is not yet available is
zero-padded, and :meth:`~mushroom_rl.core.history_manager.HistoryManager.reset` clears the buffers at the start of
an episode, so the first windows are zero-padded from the left.

.. literalinclude:: code/history_manager.py
   :lines: 19-27

Running this prints the windows growing step by step. Note the effect of the ``offset``: at step 3 the
``obs_history`` window ends at the current observation (``offset`` 0), whereas the ``action_history`` window ends
one step behind (``offset`` 1), so it holds the actions of steps 1 and 2, not the action about to be taken.

Reconstructing offline
----------------------

The same stacking rule is exposed offline through
:meth:`~mushroom_rl.core.history_manager.HistoryManager.build_history`, which rebuilds one window per timestep
from a stored buffer, walking backwards from each anchor up to the stream length and stopping at episode
boundaries (given by the ``last`` flags) or at the start of the buffer, zero-padding the missing older entries.
Feeding it the transition columns of a dataset reproduces *exactly* the windows assembled online:

.. literalinclude:: code/history_manager.py
   :lines: 29-38

The offline ``obs_history`` and ``action_history`` windows match the online ones step for step. The agent injects
the manager into the replay memory (via :attr:`~mushroom_rl.core.Agent.history_manager`), and the memory rebuilds
the stacked context for the sampled transitions with the same rule used to collect them, without ever storing the
redundant stacked windows. The circular replay-buffer variant,
:meth:`~mushroom_rl.core.history_manager.HistoryManager.build_history_circular_buffer`, does the same for a
wrapped-around buffer, taking positions modulo the buffer size and stopping at the write head.

Sequence vs. window
-------------------

The window and the sequence are two distinct concepts that should not be confused:

- the **window** is the history-stacking length of a *single* timestep — ``history_length`` observations (or
  ``action_history_length`` actions) glued together to form the input at that step. This is what the history
  manager builds.
- the **sequence** is the ``truncation_length``: a run of *consecutive* timesteps sampled together, used to train
  recurrent agents through backpropagation through time (see
  :class:`~mushroom_rl.rl_utils.replay_memory.SequenceReplayMemory` and ``PPO_BPTT``). This is a property of how
  the replay memory samples, not of the history manager.

The two act on separate axes and **compose orthogonally**. With a ``history_length`` greater than 1, each
timestep of a sampled sequence is itself a stacked window, so the sampled states have shape
``(n_samples, truncation_length, history_length, *obs_shape)``, collapsing to
``(n_samples, truncation_length, *obs_shape)`` when no observation stacking is used. The sequence axis is
contributed by the replay memory and the window axis by the history manager, and each may be enabled
independently of the other.