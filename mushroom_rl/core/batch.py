from dataclasses import dataclass, fields, KW_ONLY
from typing import Any


batch_dataclass = dataclass(slots=True, eq=False, frozen=True)


@batch_dataclass
class Batch:
    """
    Batch of transitions.

    Iterating over a batch yields the six transition arrays ``state, action, reward, next_state, absorbing, last``,
    so a batch can be unpacked as ``state, action, reward, next_state, absorbing, last = batch``. Every other field is
    accessed by name.

    The fields of a batch cannot be reassigned, while the arrays they hold can be modified in place.

    """
    state: Any
    action: Any
    reward: Any
    next_state: Any
    absorbing: Any
    last: Any
    _: KW_ONLY
    action_history: Any = None
    next_action_history: Any = None
    policy_state: Any = None
    next_policy_state: Any = None

    def __iter__(self):
        return iter((self.state, self.action, self.reward, self.next_state, self.absorbing, self.last))

    @classmethod
    def from_batch(cls, batch, **values):
        """
        Build a batch of this class holding the fields of another batch and the given ones.

        Args:
            batch (Batch): the batch whose fields are copied;
            **values: the values of the fields of this class that ``batch`` does not hold.

        Returns:
            The new batch, sharing the arrays of ``batch``.

        """
        return cls(**{f.name: getattr(batch, f.name) for f in fields(batch)}, **values)


@batch_dataclass
class PrioritizedBatch(Batch):
    """
    Batch of transitions sampled with priorities, holding the buffer positions ``idxs`` of the sampled transitions
    and their importance sampling ``weights``.

    """
    _: KW_ONLY
    idxs: Any = None
    weights: Any = None


@batch_dataclass
class SequenceBatch(Batch):
    """
    Batch of sequences. ``state``, ``action``, ``next_state``, ``action_history`` and ``next_action_history`` hold one
    zero-padded sequence per sample, with the time axis after the sample axis; ``reward``, ``absorbing`` and ``last``
    hold the value of the final step of each sequence; ``policy_state`` and ``next_policy_state`` hold the policy
    state and policy next state of the first step of each sequence; ``lengths`` holds the number of steps of each
    sequence, on the CPU.

    """
    _: KW_ONLY
    lengths: Any = None
