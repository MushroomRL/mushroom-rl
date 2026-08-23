import torch
from mushroom_rl.utils.episodes import split_episodes, unsplit_episodes


def _next_action_history(action_history, action, last):
    """
    Build the previous-action window aligned to the next state from the one aligned to the current state.

    Args:
        action_history (torch.Tensor): the previous-action window aligned to each state;
        action (torch.Tensor): the action taken at each of those steps;
        last (torch.Tensor): the episode-boundary flag of each step.

    Returns:
        The previous-action window aligned to the next state, in the same shape as ``action_history``.

    """
    if action_history.dim() == action.dim():
        return action

    boundary = last > 0
    boundary[-1] = True

    next_action_history = torch.empty_like(action_history)
    next_action_history[:-1] = action_history[1:]
    next_action_history[boundary] = torch.cat([action_history[boundary][:, 1:], action[boundary][:, None]], dim=1)

    return next_action_history


def compute_advantage_montecarlo(V, s, ss, r, absorbing, last, gamma, action_history=None, action=None):
    """
    Function to estimate the advantage and new value function target
    over a dataset. The value function is estimated using rollouts
    (Monte Carlo estimation).

    Args:
        V (Regressor): the current value function regressor;
        s (torch.tensor): the set of states in which we want to evaluate the advantage;
        ss (torch.tensor): the set of next states in which we want to evaluate the advantage;
        r (torch.tensor): the reward obtained in each transition from state s to state ss;
        absorbing (torch.tensor): an array of boolean flags indicating if the reached state is absorbing;
        gamma (float): the discount factor of the considered problem;
        action_history (torch.tensor, None): the previous-action window aligned to ``s``, forwarded to ``V`` when the
            critic conditions on it;
        action (torch.tensor, None): the action taken at each step, used to derive the previous-action window aligned to
            ``ss`` when the critic conditions on it.
    Returns:
        The new estimate for the value function of the next state
        and the advantage function.
    """
    with torch.no_grad():
        r = r.squeeze()
        v = V(s, action_history=action_history).squeeze()

        next_action_history = _next_action_history(action_history, action, last) if action_history is not None \
            else None
        arrays = (r, absorbing, ss) if next_action_history is None else (r, absorbing, ss, next_action_history)
        r_ep, absorbing_ep, ss_ep, *rest = split_episodes(last, *arrays)
        next_ah_ep = rest[0] if rest else None
        q_ep = torch.zeros_like(r_ep, dtype=torch.float32)
        step_axis = r_ep.dim() - 1
        next_ah = next_ah_ep.select(step_axis, -1) if next_ah_ep is not None else None
        q_next_ep = V(ss_ep.select(step_axis, -1), action_history=next_ah).squeeze()

        for rev_k in range(r_ep.shape[-1]):
            k = r_ep.shape[-1] - rev_k - 1
            q_next_ep = r_ep[..., k] + gamma * q_next_ep * (1 - absorbing_ep[..., k].int())
            q_ep[..., k] = q_next_ep

        q = unsplit_episodes(last, q_ep)
        adv = q - v

        return q[:, None], adv[:, None]


def compute_advantage(V, s, ss, r, absorbing, last, gamma, action_history=None, action=None):
    """
    Function to estimate the advantage and new value function target
    over a dataset. The value function is estimated using bootstrapping.

    Args:
        V (Regressor): the current value function regressor;
        s (torch.tensor): the set of states in which we want to evaluate the advantage;
        ss (torch.tensor): the set of next states in which we want to evaluate the advantage;
        r (torch.tensor): the reward obtained in each transition from state s to state ss;
        absorbing (torch.tensor): an array of boolean flags indicating if the reached state is absorbing;
        last (torch.tensor): an array of boolean flags indicating if the reached state is the last of the trajectory;
        gamma (float): the discount factor of the considered problem;
        action_history (torch.tensor, None): the previous-action window aligned to ``s``, forwarded to ``V`` when the
            critic conditions on it;
        action (torch.tensor, None): the action taken at each step, used to derive the previous-action window aligned to
            ``ss`` when the critic conditions on it.
    Returns:
        The new estimate for the value function of the next state and the advantage function.
    """
    with torch.no_grad():
        next_action_history = _next_action_history(action_history, action, last) if action_history is not None \
            else None
        v = V(s, action_history=action_history).squeeze()
        v_next = V(ss, action_history=next_action_history).squeeze() * (1 - absorbing.int())

        q = r + gamma * v_next
        adv = q - v
        return q[:, None], adv[:, None]


def compute_gae(V, s, ss, r, absorbing, last, gamma, lam, action_history=None, action=None):
    """
    Function to compute Generalized Advantage Estimation (GAE)
    and new value function target over a dataset.

    "High-Dimensional Continuous Control Using Generalized
    Advantage Estimation".
    Schulman J. et al. 2016.

    Args:
        V (Regressor): the current value function regressor;
        s (torch.tensor): the set of states in which we want to evaluate the advantage;
        ss (torch.tensor): the set of next states in which we want to evaluate the advantage;
        r (torch.tensor): the reward obtained in each transition from state s to state ss;
        absorbing (torch.tensor): an array of boolean flags indicating if the reached state is absorbing;
        last (torch.tensor): an array of boolean flags indicating if the reached state is the last of the trajectory;
        gamma (float): the discount factor of the considered problem;
        lam (float): the value for the lamba coefficient used by GEA algorithm;
        action_history (torch.tensor, None): the previous-action window aligned to ``s``, forwarded to ``V`` when the
            critic conditions on it;
        action (torch.tensor, None): the action taken at each step, used to derive the previous-action window aligned to
            ``ss`` when the critic conditions on it.
    Returns:
        The new estimate for the value function of the next state
        and the estimated generalized advantage.
    """
    with torch.no_grad():
        next_action_history = _next_action_history(action_history, action, last) if action_history is not None \
            else None
        v = V(s, action_history=action_history)
        v_next = V(ss, action_history=next_action_history)

        v_ep, v_next_ep, r_ep, absorbing_ep = split_episodes(last, v.squeeze(), v_next.squeeze(), r, absorbing)
        gen_adv_ep = torch.zeros_like(v_ep)
        for rev_k in range(v_ep.shape[-1]):
            k = v_ep.shape[-1] - rev_k - 1
            if rev_k == 0:
                gen_adv_ep[..., k] = r_ep[..., k] - v_ep[..., k] + \
                                     (1 - absorbing_ep[..., k].int()) * gamma * v_next_ep[..., k]
            else:
                gen_adv_ep[..., k] = r_ep[..., k] - v_ep[..., k] + \
                                     (1 - absorbing_ep[..., k].int()) * gamma * v_next_ep[..., k] + \
                                     gamma * lam * gen_adv_ep[..., k + 1]

        gen_adv = unsplit_episodes(last, gen_adv_ep).unsqueeze(-1)

        return gen_adv + v, gen_adv
