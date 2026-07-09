import numpy as np

from mushroom_rl.core import MDPInfo, AgentInfo, Box
from mushroom_rl.core.history_manager import HistoryManager

# MDP and agent info
observation_space = Box(low=-1., high=1., shape=(2,))
action_space = Box(low=-1., high=1., shape=(1,))
mdp_info = MDPInfo(observation_space, action_space, gamma=0.99, horizon=100)
agent_info = AgentInfo(is_episodic=False, policy_state_shape=None, backend='numpy')

# History manager
history = HistoryManager.from_infos(mdp_info, agent_info, history_length=3, action_history_length=2)

print('history_length:', history.history_length)
print('action_history_length:', history.action_history_length)
print('max_reach:', history.max_reach)

# Online stacking
history.reset()
prev_action = np.zeros(1)
for t in range(4):
    obs = np.full(2, t, dtype=float)
    state, policy_kwargs = history(obs, action_history=prev_action)
    print(f'--- step {t} ---')
    print('obs window:\n', state)
    print('action_history window:\n', policy_kwargs['action_history'])
    prev_action = np.full(1, t)

# Offline reconstruction
states = np.stack([np.full(2, t, dtype=float) for t in range(4)])
actions = np.stack([np.full(1, t, dtype=float) for t in range(4)])
last = np.zeros(4, dtype=bool)

obs_windows = history.build_history('obs', states, last)
action_windows = history.build_history('action_history', actions, last)
print('=== offline ===')
print('obs windows:\n', obs_windows)
print('action_history windows:\n', action_windows)