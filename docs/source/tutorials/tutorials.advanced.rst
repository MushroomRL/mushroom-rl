How to make an advanced experiment
==================================

Continuous MDPs are a challenging class of problems to solve in RL. In these
problems, a tabular regressor is not enough to approximate the Q-function, since
there are an infinite number of states/actions. The solution to solve them is to
use a function approximator (e.g. neural network) fed with the raw values
of states and actions. In the case a linear approximator is used, it is
convenient to enlarge the input space with the space of non-linear **features**
extracted from the raw values. This way, the linear approximator is often able
to solve the MDPs, despite its simplicity. Many RL algorithms rely on the use of
a linear approximator to solve a MDP, therefore the use of features is very
important.
This tutorial shows how to solve a continuous MDP in Mushroom using an
algorithm that requires the use of a linear approximator.

Initially, the MDP is created:

::

    # MDP
    mdp = Gym(name='MountainCar-v0', horizon=np.inf, gamma=1.)

This is an environment created with the Mushroom interface to the OpenAI Gym
library. Each environment offered by OpenAI Gym can be created this way simply
providing the corresponding id in the ``name`` parameter, except for the Atari
that are managed by a separate class.
After the creation of the MDP, the features are created:

::

    # Agent
    tilings = Tiles.generate(10, [10, 10],
                             mdp.info.observation_space.low,
                             mdp.info.observation_space.high)
    features = Features(tilings=tilings)

    ...

    agent = TrueOnlineSARSALambda(pi, mdp.info, agent_params, features)

In this example, we use sparse coding by means of **tiles** features. The
``generate`` method generates the grid of tilings with the required dimension
(the way the tilings are created is explained in *"Reinforcement Learning: An Introduction",
Sutton & Barto, 1998*). Eventually, the grid is passed to the ``Features``
factory method that returns the features that are fed to the algorithm.
Doing this, at each call of the Q-regressor, the algorithm will preprocess its
input extracting the features that are supposed to be fed into it.

Mushroom offers other type of features such a **radial basis functions** and
**polynomial** features. The former have also a faster implementation written in
Tensorflow that can be used transparently.
