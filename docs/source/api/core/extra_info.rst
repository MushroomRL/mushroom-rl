Extra info
==========

``ExtraInfo`` holds everything a dataset stores beside the transitions and keeps its parts in step with each
other: ``StepInfo`` turns the per-step information an environment returns into a flat dictionary of arrays, and
``EpisodeInfo`` holds one entry per episode for every environment, both for what an environment reports when it
resets and for the policy parameters an episodic agent draws.

.. automodule:: mushroom_rl.core.extra_info
