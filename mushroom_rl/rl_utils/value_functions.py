import torch
from mushroom_rl.utils.episodes import split_episodes, unsplit_episodes

def compute_advantage_montecarlo(V, s, ss, r, absorbing, last, gamma):
    """
    Function to estimate the advantage and new value function target
    over a dataset. The value function is estimated using rollouts
    (Monte Carlo estimation).

    Args:
        V (Regressor): the current value function regressor;
        s (torch.tensor): the set of states in which we want
            to evaluate the advantage;
        ss (torch.tensor): the set of next states in which we want
            to evaluate the advantage;
        r (torch.tensor): the reward obtained in each transition
            from state s to state ss;
        absorbing (torch.tensor): an array of boolean flags indicating
            if the reached state is absorbing;
        gamma (float): the discount factor of the considered problem.
    Returns:
        The new estimate for the value function of the next state
        and the advantage function.
    """
    with torch.no_grad():
        r = r.squeeze()
        v = V(s).squeeze()

        r_ep, absorbing_ep, ss_ep = split_episodes(last, r, absorbing, ss)
        q_ep = torch.zeros_like(r_ep, dtype=torch.float32)
        q_next_ep = V(ss_ep[..., -1, :]).squeeze()

        for rev_k in range(r_ep.shape[-1]):
            k = r_ep.shape[-1] - rev_k - 1
            q_next_ep = r_ep[..., k] + gamma * q_next_ep * (1 - absorbing_ep[..., k].int())
            q_ep[..., k] = q_next_ep

        q = unsplit_episodes(last, q_ep)
        adv = q - v

        return q[:, None], adv[:, None]

def compute_advantage(V, s, ss, r, absorbing, gamma):
    """
    Function to estimate the advantage and new value function target
    over a dataset. The value function is estimated using bootstrapping.

    Args:
        V (Regressor): the current value function regressor;
        s (torch.tensor): the set of states in which we want
            to evaluate the advantage;
        ss (torch.tensor): the set of next states in which we want
            to evaluate the advantage;
        r (torch.tensor): the reward obtained in each transition
            from state s to state ss;
        absorbing (torch.tensor): an array of boolean flags indicating
            if the reached state is absorbing;
        gamma (float): the discount factor of the considered problem.
    Returns:
        The new estimate for the value function of the next state
        and the advantage function.
    """
    with torch.no_grad():
        v = V(s).squeeze()
        v_next = V(ss).squeeze() * (1 - absorbing.int())

        q = r + gamma * v_next
        adv = q - v
        return q[:, None], adv[:, None]


def compute_gae(V, s, ss, r, absorbing, last, gamma, lam):
    """
    Function to compute Generalized Advantage Estimation (GAE)
    and new value function target over a dataset.

    "High-Dimensional Continuous Control Using Generalized
    Advantage Estimation".
    Schulman J. et al.. 2016.

    Args:
        V (Regressor): the current value function regressor;
        s (torch.tensor): the set of states in which we want
            to evaluate the advantage;
        ss (torch.tensor): the set of next states in which we want
            to evaluate the advantage;
        r (torch.tensor): the reward obtained in each transition
            from state s to state ss;
        absorbing (torch.tensor): an array of boolean flags indicating
            if the reached state is absorbing;
        last (torch.tensor): an array of boolean flags indicating
            if the reached state is the last of the trajectory;
        gamma (float): the discount factor of the considered problem;
        lam (float): the value for the lamba coefficient used by GEA
            algorithm.
    Returns:
        The new estimate for the value function of the next state
        and the estimated generalized advantage.
    """
    with torch.no_grad():
        v = V(s)
        v_next = V(ss)

        v_ep, v_next_ep, r_ep, absorbing_ep = split_episodes(last, v.squeeze(), v_next.squeeze(), r, absorbing)
        gen_adv_ep = torch.zeros_like(v_ep)
        for rev_k in range(v_ep.shape[-1]):
            k = v_ep.shape[-1] - rev_k - 1
            if rev_k == 0:
                gen_adv_ep[..., k] = r_ep[..., k] - v_ep[..., k] + (1 - absorbing_ep[..., k].int()) * gamma * v_next_ep[..., k]
            else:
                gen_adv_ep[..., k] = r_ep[..., k] - v_ep[..., k] + (1 - absorbing_ep[..., k].int()) * gamma * v_next_ep[..., k] + gamma * lam * gen_adv_ep[..., k + 1]

        gen_adv = unsplit_episodes(last, gen_adv_ep).unsqueeze(-1)

        return gen_adv + v, gen_adv