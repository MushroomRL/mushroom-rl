import dataclasses
import pickle

import numpy as np
import pytest

from mushroom_rl.core import Batch, PrioritizedBatch, SequenceBatch


def test_batch():
    transition = (np.zeros((3, 2)), np.zeros((3, 1)), np.ones(3), np.ones((3, 2)), np.zeros(3, dtype=bool),
                  np.ones(3, dtype=bool))
    batch = Batch(*transition)

    assert all(x is y for x, y in zip(batch, transition)) and len(list(batch)) == 6
    assert batch.action_history is None and batch.next_action_history is None
    assert batch.policy_state is None and batch.next_policy_state is None

    batch.state[0, 0] = 5.
    assert transition[0][0, 0] == 5.
    with pytest.raises(dataclasses.FrozenInstanceError):
        batch.state = np.ones((3, 2))
    with pytest.raises((TypeError, AttributeError)):
        batch.stat = np.ones((3, 2))

    loaded = pickle.loads(pickle.dumps(batch))
    assert all(np.array_equal(x, y) for x, y in zip(loaded, batch))


def test_batch_classes():
    transition = tuple(np.arange(3) + i for i in range(6))
    batch = Batch(*transition, action_history=np.ones((3, 2)), policy_state=np.zeros((3, 4)))
    prioritized = PrioritizedBatch.from_batch(batch, idxs=np.array([4, 1, 2]), weights=np.ones(3))
    sequence = SequenceBatch(*transition, lengths=np.array([2, 3, 1]))

    assert type(prioritized) is PrioritizedBatch and len(list(prioritized)) == 6 and len(list(sequence)) == 6
    assert all(x is y for x, y in zip(prioritized, batch))
    assert prioritized.action_history is batch.action_history and prioritized.policy_state is batch.policy_state
    assert np.array_equal(prioritized.idxs, np.array([4, 1, 2])) and np.array_equal(sequence.lengths, [2, 3, 1])
    for cls in (Batch, PrioritizedBatch, SequenceBatch):
        assert '__slots__' in cls.__dict__ and '__eq__' not in cls.__dict__
        assert dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen
