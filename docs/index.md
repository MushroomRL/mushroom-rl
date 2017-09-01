# Mushroom
## Reinforcement Learning python library

Mushroom is a Reinforcement Learning (RL) library that aims to be a simple, yet
powerful way to make RL experiments in a fast way. The idea behind Mushroom consists
in offering the majority of RL algorithms providing a common interface
in order to run them without excessive effort. It makes a large use of the environments
provided by [OpenAI Gym](https://gym.openai.com/) library and of the regression models
provided by [Scikit-Learn](http://scikit-learn.org/stable/) library giving also the possibility
to build and run neural networks using [Tensorflow](https://www.tensorflow.org) library.

With Mushroom you can:

* Solve value-based RL problems simply writing a single small script.
* Use all RL environments offered by OpenAI Gym and build customized environments as well.
* Exploit regression models offered by Scikit-Learn or build a customized one with Tensorflow.
* Run experiments with CPU or GPU.

## Basic run example
### Solve a discrete MDP in few a lines
Firstly, create a `mdp`:

    import mushroom
    from mushroom.environments.grid_world import GridWorld

    mdp = GridWorld(width=3, height=3, goal=(2, 2), start=(0, 0))
    
Then, an `epsilon`-greedy policy `pi` with:

    from mushroom.policy.policy import EpsGreedy
    from mushroom.utils.parameters import Parameter 

    epsilon = Parameter(value=1)
    pi = EpsGreedy(epsilon=epsilon, observation_space=mdp.observation_space,
                   action_space=mdp.action_space)
                   
The Q-function `approximator` is created with:

    from mushroom.approximators import *
 
    shape = mdp.observation_space.size + mdp.action_space.size
    approximator_params = dict(shape=shape)
    approximator = Regressor(Tabular,
                             discrete_actions=mdp.action_space.n,
                             **approximator_params)
                             
Eventually, the `agent` is:

    from mushroom.algorithms.td import QLearning 
                             
    learning_rate = Parameter(value=.6)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = QLearning(approximator, pi, mdp.gamma, **agent_params)
    
Learn:

    from mushroom.core.core import Core

    core = Core(agent, mdp)
    core.learn(n_iterations=10000, how_many=1, n_fit_steps=1,
               iterate_over='samples')
               
Final Q-table:

    import numpy as np
    q = np.zeros(shape)
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            for k in xrange(shape[2]):
                state = np.array([[i, j]])
                action = np.array([[k]])
                q[i, j, k] = approximator.predict([state, action])
    print(q)
    
    '''
    Results in:
    [[[  6.561        7.29         6.561        7.29      ]
      [  7.29         8.1          6.561        8.06685386]
      [  8.1          9.           7.29         8.1       ]]
    
     [[  6.561        8.00701093   7.29         8.1       ]
      [  7.29         9.           7.29         9.        ]
      [  8.1         10.           8.1          9.        ]]
    
     [[  7.29         8.1          8.1          9.        ]
      [  8.1          9.           8.1         10.        ]
      [  0.           0.           0.           0.        ]]]
    
    where dimensions are width, height and actions.
    """

## Installation


