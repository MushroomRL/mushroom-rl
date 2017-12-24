How to make an experiment
=========================

The main purpose of Mushroom is to simplify the scripting of RL experiments. A
standard example of a script to run an experiment in Mushroom, consists of:

* an **initial part** where the setting of the experiment are set;
* a **middle part** where the experiment is run;
* a **final part** where operations like evaluation, plot and save can be done.

A RL experiment consists of:

* a **MDP**;
* an **agent**;
* the **core**.

A **MDP** is the problem to be solved by the agent. It contains the function to move
the agent in the environment according to the provided action.
The MDP can be simply created with:

::

    mdp = CarOnHill()

A Mushroom **agent** is the algorithm that is run to learn in the MDP. It consists
of a policy approximator and of the methods to improve the policy during the
learning. It also contains the features to extract in the case of MDP with continuous
state and action spaces. An agent can be defined this way:

::

    # Policy
    epsilon = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon)

    # Approximator
    approximator_params = dict(input_shape=mdp.info.observation_space.shape,
                               n_actions=mdp.info.action_space.n,
                               n_estimators=50,
                               min_samples_split=5,
                               min_samples_leaf=2)
    approximator = ExtraTreesRegressor

    # Agent
    algorithm_params = dict(n_iterations=20)
    fit_params = dict()
    agent_params = {'approximator_params': approximator_params,
                    'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = FQI(approximator, pi, mdp.info, agent_params)

This piece of code creates the policy followed by the agent (e.g. :math:`\epsilon`-greedy)
with :math:`\epsilon = 1`. Then, the policy approximator is created specifying the
parameters to create it and the class (in this case, the ``ExtraTreesRegressor`` class
of scikit-learn is used). Eventually, the agent is created calling the algorithm
class and providing the approximator and the policy, together with parameters used
by the algorithm.

To run the experiment, the **core** module has to be used. This module requires
the agent and the MDP object and contains the function to learn in the MDP and
evaluate the learned policy. It can be created with:

::

    core = Core(agent, mdp)

Once the core has been created, the agent can be trained collecting a dataset and
fitting the policy:

::

    core.learn(n_episodes=1000, n_episodes_per_fit=1000)

In this case, the agent's policy is fitted only once, after that 1000 episodes
have been collected. This is a common practice in batch RL algorithms such as
``FQI`` where, initially, samples are randomly collected and then the policy is fitted
using the whole dataset of collected samples. In the case of Temporal-Difference (TD) algorithms,
typically the training is done with:

::

    core.learn(n_samples=1000, n_samples_per_fit=1)

This way, 1000 samples are collected, but the policy is fitted after each sample.

More unusual operations like:

::

    core.learn(n_samples=1000, n_episodes_per_fit=1)

are allowed.

Eventually, some operations to evaluate the learned policy can be done. The way to do
this is to run:

::

    dataset = core.evaluate(n_samples=1000)

This function returns the dataset collected running the learned policy after 1000
samples. This way the user can, for instance, compute the performance of the agent
through the collected rewards.