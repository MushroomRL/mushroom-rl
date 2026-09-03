import numpy as np

from mushroom_rl.core import MDPInfo, AgentInfo, Box, Dataset
from mushroom_rl.core.history_manager import HistoryManager
from mushroom_rl.rl_utils.preprocessors import StandardizationPreprocessor

# MDP and agent info
observation_space = Box(low=-1., high=1., shape=(2,))
action_space = Box(low=-1., high=1., shape=(1,))
mdp_info = MDPInfo(observation_space, action_space, gamma=0.99, horizon=100)
agent_info = AgentInfo(is_episodic=False, policy_state_shape=None, backend='numpy')

# History manager
history = HistoryManager.default_streams(mdp_info, agent_info, history_length=3, action_history_length=2)

print('history_length:', history.history_length)
print('action_history_length:', history.action_history_length)
print('max_reach:', history.max_reach)

# Sample dataset, used below both to drive the online stacking and for the offline reconstruction
states = np.stack([np.full(2, t, dtype=float) for t in range(4)])
actions = np.stack([np.full(1, t, dtype=float) for t in range(4)])
rewards = np.arange(4, dtype=float)
next_states = np.stack([np.full(2, t, dtype=float) for t in range(1, 5)])
absorbing = np.zeros(4, dtype=bool)
last = np.zeros(4, dtype=bool)
dataset = Dataset.from_array(states, actions, rewards, next_states, absorbing, last,
                             horizon=100, gamma=0.99, backend='numpy')

# Online stacking
print('#' * 40)
print('online')
history.reset()
for t in range(len(dataset)):
    obs = dataset.state[t]  # simulates the observation returned by env.step() at this timestep
    state, policy_kwargs = history(obs)
    print('-' * 40)
    print(f'step {t}:')
    print('obs window:\n', state)
    print('action_history window:\n', policy_kwargs['action_history'])
    history.record_action(dataset.action[t])

# Offline reconstruction
obs_windows = history.build_history('obs_history', dataset.state, dataset.last)
action_windows = history.build_history('action_history', dataset.action, dataset.last)
print('#' * 40)
print('offline')
for t in range(len(dataset)):
    print('-' * 40)
    print(f'step {t}:')
    print('obs window:\n', obs_windows[t])
    print('action_history window:\n', action_windows[t])

# Parsing a whole dataset at once
state, action, reward, next_state, absorbing, last, extra = history.parse_history(dataset)
print('#' * 40)
print('parse_history')
for t in range(len(dataset)):
    print('-' * 40)
    print(f'step {t}:')
    print('state window:\n', state[t])
    print('action_history window:\n', extra['action_history'][t])

# n-step return folded into the parse
_, _, nstep_reward, nstep_next_state, nstep_absorbing, nstep_last, nstep_extra = \
    history.parse_nstep_history(dataset, gamma=0.9, n_steps_return=2)
print('#' * 40)
print('parse_nstep_history (n_steps_return=2)')
print('n-step reward:\n', nstep_reward)
print('endpoint:\n', nstep_extra['endpoint'])

# Observation preprocessing, applied before the stacking
preprocessor = StandardizationPreprocessor(mdp_info)
history.add_preprocessor(preprocessor)
history.update_preprocessors(dataset)
print('#' * 40)
print('preprocessing')
print('statistics mean:\n', preprocessor._obs_runstand.mean)
print('preprocessed state window:\n', history.parse_state(dataset)[-1])
print('initial state window:\n', history.parse_initial_state(dataset))
