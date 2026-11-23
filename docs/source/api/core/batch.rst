Batch
=====

A ``Batch`` holds the arrays of a set of transitions. It is returned by
:meth:`Dataset.parse <mushroom_rl.core.dataset.Dataset.parse>`, by
:meth:`~mushroom_rl.core.history_manager.HistoryManager.parse_history` and
:meth:`~mushroom_rl.core.history_manager.HistoryManager.parse_nstep_history`, and by the ``get`` method of the replay
memories.

Iterating over a batch yields the six transition arrays, so every batch unpacks in the same way. Every other field
is read by name:

.. code-block:: python

    batch = replay_memory.get(batch_size)
    state, action, reward, next_state, absorbing, last = batch
    previous_actions = batch.action_history

The fields of a batch cannot be reassigned, while the arrays they hold can be modified in place. The arrays may share
memory with each other and with the dataset they are parsed from:

.. code-block:: python

    batch.state[:] = 0   # modifies the array
    batch.state = other  # raises dataclasses.FrozenInstanceError
    batch.state += 1     # modifies the array, then raises dataclasses.FrozenInstanceError

Fields
------

The fields of a batch of ``n`` transitions are listed in the following table.

.. csv-table::
   :header: "Field", "Content", "Shape"
   :widths: 22, 48, 30

   "``state``, ``next_state``", "The observation, or the window of the ``H`` most recent observations when the history length ``H`` is greater than 1.", "``(n, *obs_shape)``, or ``(n, H, *obs_shape)``"
   "``action``", "The action.", "``(n, *action_shape)``"
   "``reward``", "The reward.", "``(n,)``"
   "``absorbing``, ``last``", "The absorbing flag and the episode/segment end flag.", "``(n,)``"
   "``action_history``, ``next_action_history``", "The window of the ``L`` previous actions of the step and of the next step, ``None`` when the previous-action stream is not active.", "``(n, L, *action_shape)``, or ``(n, *action_shape)`` when ``L`` is 1"
   "``policy_state``, ``next_policy_state``", "The policy state before and after the step, ``None`` unless requested.", "``(n, *policy_state_shape)``"
   "``critic_state``, ``critic_next_state``", "The critic input of the step and of the next step, the same arrays as ``state`` and ``next_state`` unless the critic reads another input.", "``(n, *critic_shape)``"

The policy state is requested with ``get_policy_state=True``, or by building a replay memory with
``store_policy_state=True``. The arrays of each kind the batch holds are given by the properties in the following
table.

.. csv-table::
   :header: "Property", "Arrays"
   :widths: 30, 70

   "``state_arrays``", "``state``, ``next_state``, and ``critic_state``, ``critic_next_state`` when the batch holds a critic input"
   "``action_arrays``", "``action``, and ``action_history``, ``next_action_history`` when present"
   "``policy_state_arrays``", "``policy_state``, ``next_policy_state`` when present"

Critic input
------------

An algorithm reads its critic input from ``critic_state`` and ``critic_next_state`` unconditionally: they differ from
``state`` and ``next_state`` only when the critic reads other observation entries, or another history length, than
the actor.

.. code-block:: python

    batch = replay_memory.get(batch_size)
    q = critic(batch.critic_state, batch.action)

n-step batches
--------------

In a batch of n-step transitions, returned by
:meth:`~mushroom_rl.core.history_manager.HistoryManager.parse_nstep_history` and by a replay memory built with
``n_steps_return`` greater than 1, every row spans the ``n`` steps of the return. ``state``, ``action``,
``action_history`` and ``policy_state`` are those of its first step, ``reward`` is the discounted return, and
``next_state``, ``critic_next_state``, ``next_action_history``, ``next_policy_state``, ``absorbing`` and ``last`` are
taken at the final step of the return. The shapes are the ones of the table above.

Prioritized batches
-------------------

:class:`~mushroom_rl.core.batch.PrioritizedBatch` is returned by
:class:`~mushroom_rl.rl_utils.replay_memory.PrioritizedReplayMemory`. It adds the buffer positions ``idxs`` of the
sampled transitions and their importance sampling ``weights``, both of shape ``(n,)``. The positions are passed back
with the new errors to update the priorities:

.. code-block:: python

    batch = replay_memory.get(batch_size)
    ...
    replay_memory.update(td_error, batch.idxs)

Sequence batches
----------------

:class:`~mushroom_rl.core.batch.SequenceBatch` is returned by
:class:`~mushroom_rl.rl_utils.replay_memory.SequenceReplayMemory` and holds one sequence of at most ``T`` consecutive
steps per sample, ``T`` being the truncation length. What each field holds is listed in the following table.

.. csv-table::
   :header: "Fields", "Content", "Shape"
   :widths: 40, 30, 30

   "``state``, ``action``, ``next_state``, ``action_history``, ``next_action_history``, ``critic_state``, ``critic_next_state``", "The whole sequence, zero-padded after its final step.", "``(n, T, ...)``"
   "``reward``, ``absorbing``, ``last``", "The value of the final step of the sequence.", "``(n,)``"
   "``policy_state``, ``next_policy_state``", "The value of the first step of the sequence.", "``(n, *policy_state_shape)``"
   "``lengths``", "The number of steps of the sequence, on the CPU.", "``(n,)``"

.. automodule:: mushroom_rl.core.batch
