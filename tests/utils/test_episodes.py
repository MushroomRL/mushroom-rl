import torch
import numpy as np

from mushroom_rl.core import Core, Agent
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.environments import Segway

from mushroom_rl.utils.episodes import split_episodes, unsplit_episodes

def test_torch_split():
    torch.manual_seed(42)
    mdp = Segway()
    state, action, reward, next_state, absorbing, last = get_episodes(mdp)

    ep_arrays = split_episodes(last, state, action, reward, next_state, absorbing, last)
    un_state, un_action, un_reward, un_next_state, un_absorbing, un_last = unsplit_episodes(last, *ep_arrays)

    assert torch.allclose(state, un_state)
    assert torch.allclose(action, un_action)
    assert torch.allclose(reward, un_reward)
    assert torch.allclose(next_state, un_next_state)
    assert torch.allclose(absorbing, un_absorbing)
    assert torch.allclose(last, un_last)

def test_numpy_split():
    torch.manual_seed(42)
    np.random.seed(42)
    
    mdp = Segway()
    state, action, reward, next_state, absorbing, last = get_episodes(mdp)

    state, action, reward, next_state, absorbing, last = state.numpy(), action.numpy(), reward.numpy(), next_state.numpy(), absorbing.numpy(), last.numpy()

    ep_arrays = split_episodes(last, state, action, reward, next_state, absorbing, last)
    un_state, un_action, un_reward, un_next_state, un_absorbing, un_last = unsplit_episodes(last, *ep_arrays)

    assert np.allclose(state, un_state)
    assert np.allclose(action, un_action)
    assert np.allclose(reward, un_reward)
    assert np.allclose(next_state, un_next_state)
    assert np.allclose(absorbing, un_absorbing)
    assert np.allclose(last, un_last)
    
def get_episodes(mdp, n_episodes=100):
    mu = np.array([6.31154476, 3.32346271, 0.49648221])
    
    approximator = LinearApproximator(input_shape=mdp.info.observation_space.shape,
                                      output_shape=mdp.info.action_space.shape,
                                      weights=mu)
                             
    policy = DeterministicPolicy(approximator)

    agent = Agent(mdp.info, policy)
    core = Core(agent, mdp)
    dataset = core.evaluate(n_episodes=n_episodes)

    return dataset.parse(to='torch')
