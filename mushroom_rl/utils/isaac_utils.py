import torch


def convert_task_observation(observation):
    obs_t = observation
    for _ in range(5):
        if torch.is_tensor(obs_t):
            break
        obs_t = obs_t[list(obs_t.keys())[0]]
    return obs_t