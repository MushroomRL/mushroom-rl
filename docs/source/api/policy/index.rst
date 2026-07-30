Policy
======

A policy defines how the agent behaves: it maps the current state to an action, either deterministically or by sampling
from a probability distribution over the action space. It is invoked by the agent through
:meth:`~mushroom_rl.core.Agent.draw_action`, and it is the object that most learning algorithms optimize.

For greedy evaluation (see :meth:`~mushroom_rl.core.Core.evaluate`), a policy also exposes
:meth:`~mushroom_rl.policy.Policy.draw_action_greedy`, returning the *mode* of the policy (e.g. the mean of a Gaussian,
the argmax of an :math:`\varepsilon`-greedy or Boltzmann policy). The base implementation raises, so a policy supports
greedy evaluation only if it overrides it. For Torch policies the greedy action comes from the underlying distribution
(a Gaussian returns its mean, a categorical its ``mode``/argmax, a squashed Gaussian its ``median``, since its true
mode is ill-behaved and can pile up at the action bounds).

Two mixins add optional capabilities that can be combined with a policy: :class:`~mushroom_rl.policy.HasWeights` equips
it with a set of trainable weights (used by policy-search and black-box optimization algorithms), while
:class:`~mushroom_rl.policy.HasGradient` additionally provides the gradient of the log-probability required by
policy-gradient methods.

MushroomRL provides several families of policies:

- **Deterministic policies** return a single action for each state;
- **Gaussian policies** are differentiable parametric policies that sample from a Gaussian distribution;
- **TD policies** are value-based policies that select the action from a Q-function (e.g. epsilon-greedy or Boltzmann);
- **Torch policies** are implemented as neural networks and support tensor computation for deep RL;
- **Movement primitives** are trajectory generators implementing DMPs and ProMPs;
- **Vector policies** wrap a population of policies for vectorized black-box optimization.

Policies in MushroomRL can depend on the past in two orthogonal ways. A **stateful policy**
(:class:`~mushroom_rl.policy.StatefulPolicy`) carries a *latent* internal state, updated at every step and stored in the
dataset because it cannot be reconstructed (e.g. a recurrent hidden state or Ornstein-Uhlenbeck noise). The **context**
is instead a deterministic function of the observed trajectory (e.g. a window of stacked observations); being
reconstructable from the stored transitions, it is assembled on the fly by the
:class:`~mushroom_rl.core.history_manager.HistoryManager` rather than stored as policy state.

.. autosummary::
   :nosignatures:

   ~mushroom_rl.policy.policy.Policy
   ~mushroom_rl.policy.policy.StatefulPolicy
   ~mushroom_rl.policy.policy.HasWeights
   ~mushroom_rl.policy.policy.HasGradient
   ~mushroom_rl.policy.td_policy.EpsGreedy
   ~mushroom_rl.policy.td_policy.Boltzmann
   ~mushroom_rl.policy.td_policy.Mellowmax
   ~mushroom_rl.policy.gaussian_policy.GaussianPolicy
   ~mushroom_rl.policy.deterministic_policy.DeterministicPolicy
   ~mushroom_rl.policy.noise_policy.OrnsteinUhlenbeckPolicy
   ~mushroom_rl.policy.noise_policy.ClippedGaussianPolicy
   ~mushroom_rl.policy.torch_policy.TorchPolicy
   ~mushroom_rl.policy.stateful_torch_policy.StatefulTorchPolicy
   ~mushroom_rl.policy.vector_policy.VectorPolicy
   ~mushroom_rl.policy.promps.ProMP
   ~mushroom_rl.policy.dmp.DMP

.. toctree::
   :maxdepth: 1

   policy
   td_policy
   gaussian_policy
   deterministic_policy
   noise_policy
   torch_policy
   stateful_torch_policy
   vector_policy
   movement_primitives
