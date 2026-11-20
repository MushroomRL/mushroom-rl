Batch
=====

A ``Batch`` holds the arrays of a set of transitions. It is returned by
:meth:`Dataset.parse <mushroom_rl.core.dataset.Dataset.parse>`, by
:meth:`~mushroom_rl.core.history_manager.HistoryManager.parse_history` and
:meth:`~mushroom_rl.core.history_manager.HistoryManager.parse_nstep_history`, and by the ``get`` method of the replay
memories.

Unpacking
---------

Iterating over a batch yields the six transition arrays, so every batch unpacks in the same way. Every other field
is read by name:

.. code-block:: python

    batch = replay_memory.get(batch_size)
    state, action, reward, next_state, absorbing, last = batch
    previous_actions = batch.action_history

Fields
------

The fields of a batch of ``n`` transitions are listed in the following table.

.. csv-table::
   :header: "Field", "Content", "Shape"
   :widths: 22, 48, 30

   "``state``, ``next_state``", "The observation, or the window of the ``H`` most recent observations when the history length ``H`` is greater than 1.", "``(n, *obs_shape)``, or ``(n, H, *obs_shape)``"
   "``action``", "The action.", "``(n, *action_shape)``"
   "``reward``", "The reward, or the discounted n-step return.", "``(n,)``"
   "``absorbing``, ``last``", "The absorbing flag and the episode/segment end flag.", "``(n,)``"
   "``action_history``, ``next_action_history``", "The window of the ``L`` previous actions of the step and of the next step when the previous-action stream is active, ``None`` otherwise.", "``(n, L, *action_shape)``, or ``(n, *action_shape)`` when ``L`` is 1"
   "``policy_state``, ``next_policy_state``", "The policy state before and after the step when requested, ``None`` otherwise.", "``(n, *policy_state_shape)``"

The policy state is requested with ``get_policy_state=True``, or by building a replay memory with
``store_policy_state=True``. With an n-step return, ``next_state``, ``next_action_history``, ``absorbing``, ``last``
and ``next_policy_state`` are taken at the final step of the return.

Variants
--------

:class:`~mushroom_rl.core.batch.PrioritizedBatch` is returned by
:class:`~mushroom_rl.rl_utils.replay_memory.PrioritizedReplayMemory`. It adds the buffer positions ``idxs`` of the
sampled transitions and their importance sampling ``weights``, both of shape ``(n,)``. The positions are passed back
with the new errors to update the priorities:

.. code-block:: python

    batch = replay_memory.get(batch_size)
    state, action, reward, next_state, absorbing, last = batch
    ...
    replay_memory.update(td_error, batch.idxs)

:class:`~mushroom_rl.core.batch.SequenceBatch` is returned by
:class:`~mushroom_rl.rl_utils.replay_memory.SequenceReplayMemory` and holds one sequence of at most ``T`` steps per
sample, ``T`` being the truncation length. The per-step fields, i.e. ``state``, ``action``, ``next_state``,
``action_history`` and ``next_action_history``, have the time axis after the sample axis, e.g.
``(n, T, *obs_shape)``, and are zero-padded after the final step of a shorter sequence. ``reward``, ``absorbing`` and
``last`` hold the value of the final step of each sequence, with shape ``(n,)``. ``policy_state`` and
``next_policy_state`` hold the policy state and policy next state of the first step of each sequence, with shape
``(n, *policy_state_shape)``, and ``lengths`` is the integer array of the number of steps of each sequence, with
shape ``(n,)``, on the CPU.

Fixed fields
------------

The fields of a batch cannot be reassigned, while the arrays they hold can be modified in place. The arrays may share
memory with each other and with the dataset they are parsed from:

.. code-block:: python

    batch.state[:] = 0   # modifies the array
    batch.state = other  # raises dataclasses.FrozenInstanceError
    batch.state += 1     # modifies the array, then raises dataclasses.FrozenInstanceError

.. automodule:: mushroom_rl.core.batch
