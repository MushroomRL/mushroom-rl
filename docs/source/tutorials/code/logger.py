from mushroom_rl.core import Logger

# Create a logger object, creating a log folder
logger = Logger('tutorial', results_dir='/tmp/logs',
                log_console=True)

# Create a logger object, without creating the log folder
logger_no_folder = Logger('tutorial_no_folder', results_dir=None)

# Write a line of hashtags, to be used as a separator
logger.strong_line()

# Print an info message
logger.debug('This is a debug message')

# Print an info message
logger.info('This is an info message')

# Print a warning
logger.warning('This is a warning message')

# Print an error
logger.error('This is an error message')

# Print a critical error message
logger.critical('This is a critical error')

# Print a line of dashes, to be used as a (weak) separator
logger.weak_line()

# Exception logging
try:
    raise RuntimeError('A runtime exception occurred')
except RuntimeError as e:
    logger.error('Exception catched, here\'s the stack trace:')
    logger.exception(e)

logger.weak_line()


# Logging learning process
from mushroom_rl.core import Core
from mushroom_rl.environments.generators import generate_simple_chain
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.algorithms.value import QLearning
from mushroom_rl.rl_utils.parameters import Parameter
from tqdm import trange
from time import sleep
import numpy as np


# Setup simple learning environment
mdp = generate_simple_chain(state_n=5, goal_states=[2], prob=.8, rew=1, gamma=.9)
epsilon = Parameter(value=.15)
pi = EpsGreedy(epsilon=epsilon)
agent = QLearning(mdp.info, pi, learning_rate=Parameter(value=.2))
core = Core(agent, mdp)
epochs = 10

# Initial policy Evaluation
logger.info('Experiment started')
logger.strong_line()

dataset = core.evaluate(n_steps=100)
J = np.mean(dataset.discounted_return)
R = np.mean(dataset.undiscounted_return)  # Undiscounted returns

logger.epoch_info(0, J=J, R=R, any_label='any value')

for i in trange(epochs):
    # Here some learning
    core.learn(n_steps=100, n_steps_per_fit=1)
    sleep(0.5)
    dataset = core.evaluate(n_steps=100)
    sleep(0.5)
    J = np.mean(dataset.discounted_return)  # Discounted returns
    R = np.mean(dataset.undiscounted_return)  # Undiscounted returns

    # Here logging epoch results to the console
    logger.epoch_info(i+1, J=J, R=R)

    # Logging the data in J.npy and R.npy
    logger.log_numpy(J=J, R=R)

    # Logging the best agent according to the best J
    logger.log_best_agent(agent, J)

# Logging the last agent
logger.log_agent(agent)

# Log the last dataset
logger.log_dataset(dataset)

logger.info('Experiment terminated')

# Loggers can also continue from previous logs results
del logger  # Delete previous logger
new_logger = Logger('tutorial', results_dir='/tmp/logs',
                    log_console=True, append=True)

# add infinite at the end of J.npy
new_logger.log_numpy(J=np.inf)
new_logger.info('Tutorial ends here')
