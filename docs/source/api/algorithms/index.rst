Algorithms
==========

Every learning algorithm in MushroomRL is an ``Agent``: it is built from an ``MDPInfo`` describing the problem and,
where applicable, a ``Policy``, and it learns inside ``Agent.fit``, which ``Core`` calls with the samples collected
so far. The algorithms are grouped in three families by what they learn:

- **value-based** methods learn a value function and derive the policy from it, so they are also called
  *critic-only*;
- **policy-search** methods optimize the policy directly, either through its gradient or by black-box search over a
  distribution of policy parameters.
- **actor-critic** methods learn a policy and a value function together, using the latter to estimate the gradient
  of the former;

The tables below index every implemented algorithm. The **Actions** column says which action spaces an algorithm
supports: ``discrete`` and ``continuous`` when the algorithm constrains it, ``policy`` when the algorithm is
agnostic and it is the policy passed to the constructor that decides. The **Fit** column says what ``Agent.fit``
consumes, i.e. whether the algorithm can be fitted on a partial episode (``steps``) or requires complete episodes
(``episodes``). The **Reference** column reports the paper named in the class docstring; it is empty for the
algorithms whose docstring names none.

.. toctree::
   :maxdepth: 1

   value/index
   actor_critic/index
   policy_search/index

.. rubric:: Value-based

.. csv-table::
   :header: "Algorithm", "Class", "Actions", "Fit", "Reference"
   :widths: 20, 20, 12, 10, 38

   "Q-Learning", ":class:`~mushroom_rl.algorithms.value.td.QLearning`", "discrete", "steps", "*Learning from Delayed Rewards*. Watkins C.J.C.H. 1989."
   "Q(λ)", ":class:`~mushroom_rl.algorithms.value.td.QLambda`", "discrete", "steps", "*Learning from Delayed Rewards*. Watkins C.J.C.H. 1989."
   "Double Q-Learning", ":class:`~mushroom_rl.algorithms.value.td.DoubleQLearning`", "discrete", "steps", "*Double Q-Learning*. Hasselt H. V. 2010."
   "Weighted Q-Learning", ":class:`~mushroom_rl.algorithms.value.td.WeightedQLearning`", "discrete", "steps", "*Estimating the Maximum Expected Value through Gaussian Approximation*. D'Eramo C. et al. 2016."
   "Maxmin Q-Learning", ":class:`~mushroom_rl.algorithms.value.td.MaxminQLearning`", "discrete", "steps", "*Maxmin Q-learning: Controlling the Estimation Bias of Q-learning*. Lan Q. et al. 2019."
   "Speedy Q-Learning", ":class:`~mushroom_rl.algorithms.value.td.SpeedyQLearning`", "discrete", "steps", "*Speedy Q-Learning*. Ghavamzadeh et. al. 2011."
   "R-Learning", ":class:`~mushroom_rl.algorithms.value.td.RLearning`", "discrete", "steps", "*A Reinforcement Learning Method for Maximizing Undiscounted Rewards*. Schwartz A. 1993."
   "RQ-Learning", ":class:`~mushroom_rl.algorithms.value.td.RQLearning`", "discrete", "steps", "*Exploiting Structure and Uncertainty of Bellman Updates in Markov Decision Processes*. Tateo D. et al. 2017."
   "RQ-Learning (on-policy)", ":class:`~mushroom_rl.algorithms.value.td.RQLearningOnPolicy`", "discrete", "steps", "*Exploiting Structure and Uncertainty of Bellman Updates in Markov Decision Processes*. Tateo D. et al. 2017."
   "SARSA", ":class:`~mushroom_rl.algorithms.value.td.SARSA`", "discrete", "steps", "*On-line Q-learning using connectionist systems*. Rummery G. A. and Niranjan M. 1994"
   "SARSA(λ)", ":class:`~mushroom_rl.algorithms.value.td.SARSALambda`", "discrete", "steps", "*Reinforcement learning with replacing eligibility traces*. Singh S. P. et al. 1996."
   "SARSA(λ) continuous", ":class:`~mushroom_rl.algorithms.value.td.SARSALambdaContinuous`", "discrete", "steps", "*Reinforcement learning with replacing eligibility traces*. Singh S. P. et al. 1996."
   "Expected SARSA", ":class:`~mushroom_rl.algorithms.value.td.ExpectedSARSA`", "discrete", "steps", "*A theoretical and empirical analysis of Expected Sarsa*. Seijen H. V. et al. 2009."
   "True Online SARSA(λ)", ":class:`~mushroom_rl.algorithms.value.td.TrueOnlineSARSALambda`", "discrete", "steps", "*True Online TD(lambda)*. Seijen H. V. et al. 2014."
   "FQI", ":class:`~mushroom_rl.algorithms.value.batch_td.FQI`", "discrete", "steps", "*Tree-Based Batch Mode Reinforcement Learning*. Ernst D. et al. 2005."
   "Double FQI", ":class:`~mushroom_rl.algorithms.value.batch_td.DoubleFQI`", "discrete", "steps", "*Estimating the Maximum Expected Value in Continuous Reinforcement Learning Problems*. D'Eramo C. et al. 2017."
   "Boosted FQI", ":class:`~mushroom_rl.algorithms.value.batch_td.BoostedFQI`", "discrete", "steps", "*Boosted Fitted Q-Iteration*. Tosatto S. et al. 2017."
   "LSPI", ":class:`~mushroom_rl.algorithms.value.batch_td.LSPI`", "discrete", "steps", "*Least-Squares Policy Iteration*. Lagoudakis M. G. and Parr R. 2003."
   "DQN", ":class:`~mushroom_rl.algorithms.value.dqn.DQN`", "discrete", "steps", "*Human-Level Control Through Deep Reinforcement Learning*. Mnih V. et al. 2015."
   "Double DQN", ":class:`~mushroom_rl.algorithms.value.dqn.DoubleDQN`", "discrete", "steps", "*Deep Reinforcement Learning with Double Q-Learning*. Hasselt H. V. et al. 2016."
   "Averaged DQN", ":class:`~mushroom_rl.algorithms.value.dqn.AveragedDQN`", "discrete", "steps", "*Averaged-DQN: Variance Reduction and Stabilization for Deep Reinforcement Learning*. Anschel O. et al. 2017."
   "Categorical DQN", ":class:`~mushroom_rl.algorithms.value.dqn.CategoricalDQN`", "discrete", "steps", "*A Distributional Perspective on Reinforcement Learning*. Bellemare M. et al. 2017."
   "Dueling DQN", ":class:`~mushroom_rl.algorithms.value.dqn.DuelingDQN`", "discrete", "steps", "*Dueling Network Architectures for Deep Reinforcement Learning*. Wang Z. et al. 2016."
   "Noisy DQN", ":class:`~mushroom_rl.algorithms.value.dqn.NoisyDQN`", "discrete", "steps", "*Noisy networks for exploration*. Fortunato M. et al. 2018."
   "Quantile DQN", ":class:`~mushroom_rl.algorithms.value.dqn.QuantileDQN`", "discrete", "steps", "*Distributional Reinforcement Learning with Quantile Regression*. Dabney W. et al. 2018."
   "Maxmin DQN", ":class:`~mushroom_rl.algorithms.value.dqn.MaxminDQN`", "discrete", "steps", "*Maxmin Q-learning: Controlling the Estimation Bias of Q-learning*. Lan Q. et al. 2020."
   "Rainbow", ":class:`~mushroom_rl.algorithms.value.dqn.Rainbow`", "discrete", "steps", "*Rainbow: Combining Improvements in Deep Reinforcement Learning*. Hessel M. et al. 2018."

.. rubric:: Actor-critic

.. csv-table::
   :header: "Algorithm", "Class", "Actions", "Fit", "Reference"
   :widths: 20, 20, 12, 10, 38

   "COPDAC-Q", ":class:`~mushroom_rl.algorithms.actor_critic.classic_actor_critic.COPDAC_Q`", "continuous", "steps", "*Deterministic Policy Gradient Algorithms*. Silver D. et al. 2014."
   "Stochastic AC", ":class:`~mushroom_rl.algorithms.actor_critic.classic_actor_critic.StochasticAC`", "continuous", "steps", "*Model-Free Reinforcement Learning with Continuous Action in Practice*. Degris T. et al. 2012."
   "Stochastic AC (average reward)", ":class:`~mushroom_rl.algorithms.actor_critic.classic_actor_critic.StochasticAC_AVG`", "continuous", "steps", "*Model-Free Reinforcement Learning with Continuous Action in Practice*. Degris T. et al. 2012."
   "DDPG", ":class:`~mushroom_rl.algorithms.actor_critic.deep_actor_critic.DDPG`", "continuous", "steps", "*Continuous Control with Deep Reinforcement Learning*. Lillicrap T. P. et al. 2016."
   "TD3", ":class:`~mushroom_rl.algorithms.actor_critic.deep_actor_critic.TD3`", "continuous", "steps", "*Addressing Function Approximation Error in Actor-Critic Methods*. Fujimoto S. et al. 2018."
   "SAC", ":class:`~mushroom_rl.algorithms.actor_critic.deep_actor_critic.SAC`", "continuous", "steps", "*Soft Actor-Critic Algorithms and Applications*. Haarnoja T. et al. 2019."
   "A2C", ":class:`~mushroom_rl.algorithms.actor_critic.deep_actor_critic.A2C`", "policy", "steps", "*Asynchronous Methods for Deep Reinforcement Learning*. Mnih V. et al. 2016."
   "TRPO", ":class:`~mushroom_rl.algorithms.actor_critic.deep_actor_critic.TRPO`", "policy", "steps", "*Trust Region Policy Optimization*. Schulman J. et al. 2015."
   "PPO", ":class:`~mushroom_rl.algorithms.actor_critic.deep_actor_critic.PPO`", "policy", "steps", "*Proximal Policy Optimization Algorithms*. Schulman J. et al. 2017."
   "PPO with BPTT", ":class:`~mushroom_rl.algorithms.actor_critic.deep_actor_critic.PPO_BPTT`", "policy", "steps", "*Proximal Policy Optimization Algorithms*. Schulman J. et al. 2017."
   "Rudin PPO", ":class:`~mushroom_rl.algorithms.actor_critic.deep_actor_critic.RudinPPO`", "policy", "steps", "*Learning to walk in minutes using massively parallel deep reinforcement learning*. Rudin N. et al. 2022."

.. rubric:: Policy search

.. csv-table::
   :header: "Algorithm", "Class", "Actions", "Fit", "Reference"
   :widths: 20, 20, 12, 10, 38

   "REINFORCE", ":class:`~mushroom_rl.algorithms.policy_search.policy_gradient.REINFORCE`", "policy", "episodes", "*Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*. Williams R. J. 1992."
   "GPOMDP", ":class:`~mushroom_rl.algorithms.policy_search.policy_gradient.GPOMDP`", "policy", "episodes", "*Infinite-Horizon Policy-Gradient Estimation*. Baxter J. and Bartlett P. L. 2001."
   "eNAC", ":class:`~mushroom_rl.algorithms.policy_search.policy_gradient.eNAC`", "policy", "episodes", "*A Survey on Policy Search for Robotics*. Deisenroth M. P. et al. 2013."
   "RWR", ":class:`~mushroom_rl.algorithms.policy_search.black_box_optimization.RWR`", "policy", "episodes", "*A Survey on Policy Search for Robotics*. Deisenroth M. P. et al. 2013."
   "PGPE", ":class:`~mushroom_rl.algorithms.policy_search.black_box_optimization.PGPE`", "policy", "episodes", "*A Survey on Policy Search for Robotics*. Deisenroth M. P. et al. 2013."
   "REPS", ":class:`~mushroom_rl.algorithms.policy_search.black_box_optimization.REPS`", "policy", "episodes", "*A Survey on Policy Search for Robotics*. Deisenroth M. P. et al. 2013."
   "Constrained REPS", ":class:`~mushroom_rl.algorithms.policy_search.black_box_optimization.ConstrainedREPS`", "policy", "episodes", "*High acceleration reinforcement learning for real-world juggling with binary rewards*. Ploeger K. et al. 2020."
   "MORE", ":class:`~mushroom_rl.algorithms.policy_search.black_box_optimization.MORE`", "policy", "episodes", "*Model-Based Relative Entropy Stochastic Search*. Abdolmaleki A. et al. 2015."
   "ePPO", ":class:`~mushroom_rl.algorithms.policy_search.black_box_optimization.ePPO`", "policy", "episodes", "*Proximal Policy Optimization Algorithms*. Schulman J. et al. 2017."

The abstract bases :class:`~mushroom_rl.algorithms.value.dqn.AbstractDQN`,
:class:`~mushroom_rl.algorithms.actor_critic.deep_actor_critic.DeepAC` and
:class:`~mushroom_rl.algorithms.actor_critic.deep_actor_critic.OnPolicyDeepAC` are not listed above: they are
extension points rather than algorithms, and are documented on their own pages.
