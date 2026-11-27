import torch
from mushroom_rl.utils.episodes import episode_ends, split_episodes, unsplit_episodes


def compute_advantage_montecarlo(V, batch, gamma):
    """
    Function to estimate the advantage and new value function target
    over a dataset. The value function is estimated using rollouts
    (Monte Carlo estimation). The previous-action windows of ``batch`` are passed to ``V`` as ``action_history``.
    When the batch holds a critic input, ``V`` is evaluated on it.

    Args:
        V (Approximator): the current value function approximator;
        batch (Batch): the transitions;
        gamma (float): the discount factor of the considered problem.
    Returns:
        The new estimate for the value function of the next state
        and the advantage function.
    """
    with torch.no_grad():
        v = V(batch.critic_state, action_history=batch.action_history)
        assert v.ndim == 1, 'V must return one scalar per sample: declare output_shape=().'

        last_ep, r_ep, absorbing_ep = split_episodes(batch.last, batch.reward, batch.absorbing)
        q_ep = torch.zeros_like(r_ep, dtype=torch.float32)
        ends = episode_ends(batch.last, r_ep.shape[0] if r_ep.ndim > 1 else 1)
        next_ah = batch.next_action_history[ends] if batch.next_action_history is not None else None
        q_end = V(batch.critic_next_state[ends], action_history=next_ah)
        q_next_ep = torch.zeros_like(q_end)

        not_absorbing_ep = ~absorbing_ep

        for k in range(r_ep.shape[-1] - 1, -1, -1):
            q_next_ep = torch.where(last_ep[..., k], q_end, q_next_ep)
            q_next_ep = r_ep[..., k] + gamma * q_next_ep * not_absorbing_ep[..., k]
            q_ep[..., k] = q_next_ep

        q = unsplit_episodes(batch.last, q_ep)
        adv = q - v

        return q, adv


def compute_advantage(V, batch, gamma):
    """
    Function to estimate the advantage and new value function target
    over a dataset. The value function is estimated using bootstrapping. The previous-action windows of ``batch`` are
    passed to ``V`` as ``action_history``. When the batch holds a critic input, ``V`` is evaluated on it.

    Args:
        V (Approximator): the current value function approximator;
        batch (Batch): the transitions;
        gamma (float): the discount factor of the considered problem.
    Returns:
        The new estimate for the value function of the next state and the advantage function.
    """
    with torch.no_grad():
        v = V(batch.critic_state, action_history=batch.action_history)
        assert v.ndim == 1, 'V must return one scalar per sample: declare output_shape=().'
        v_next = V(batch.critic_next_state, action_history=batch.next_action_history) * ~batch.absorbing

        q = batch.reward + gamma * v_next
        adv = q - v
        return q, adv


def compute_gae(V, batch, gamma, lam):
    """
    Function to compute Generalized Advantage Estimation (GAE)
    and new value function target over a dataset.

    "High-Dimensional Continuous Control Using Generalized
    Advantage Estimation".
    Schulman J. et al. 2016.

    The previous-action windows of ``batch`` are passed to ``V`` as ``action_history``. When the batch holds a critic
    input, ``V`` is evaluated on it.

    Args:
        V (Approximator): the current value function approximator;
        batch (Batch): the transitions of whole trajectories, i.e. the next state of every step that does not end an
            episode is the state of the following step;
        gamma (float): the discount factor of the considered problem;
        lam (float): the value for the lamba coefficient used by GEA algorithm.
    Returns:
        The new estimate for the value function of the next state
        and the estimated generalized advantage.
    """
    with torch.no_grad():
        v = V(batch.critic_state, action_history=batch.action_history)
        assert v.ndim == 1, 'V must return one scalar per sample: declare output_shape=().'

        last_ep, v_ep, r_ep, absorbing_ep = split_episodes(batch.last, v, batch.reward, batch.absorbing)
        ends = episode_ends(batch.last, v_ep.shape[0] if v_ep.ndim > 1 else 1)
        next_ah = batch.next_action_history[ends] if batch.next_action_history is not None else None
        v_end = V(batch.critic_next_state[ends], action_history=next_ah)

        v_next_ep = torch.zeros_like(v_ep)
        v_next_ep[..., :-1] = v_ep[..., 1:]
        v_next_ep = torch.where(last_ep, v_end.reshape(-1, *(1,) * (v_ep.ndim - 1)), v_next_ep)
        delta_ep = r_ep - v_ep + ~absorbing_ep * gamma * v_next_ep

        gen_adv_ep = torch.zeros_like(v_ep)
        gen_adv_ep[..., -1] = delta_ep[..., -1]
        for k in range(v_ep.shape[-1] - 2, -1, -1):
            gen_adv_ep[..., k] = delta_ep[..., k] + gamma * lam * gen_adv_ep[..., k + 1]

        gen_adv = unsplit_episodes(batch.last, gen_adv_ep)

        return gen_adv + v, gen_adv
