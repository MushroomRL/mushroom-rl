import torch


@torch.jit.script
def torch_rand_float(lower: float, upper: float, shape: tuple[int, int], device: str) -> torch.Tensor:
    """
    Draws floats uniformly distributed in ``[lower, upper]``.

    Args:
        lower (float): The lower bound of the range drawn from.
        upper (float): The upper bound of the range drawn from.
        shape (tuple): The shape of the drawn tensor.
        device (str): The device the tensor is created on.

    Returns:
        The drawn tensor.

    """
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def quat_apply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Rotates vectors by quaternions.

    Args:
        a (torch.tensor): The quaternions, of shape ``(..., 4)`` and in ``[w, x, y, z]`` order.
        b (torch.tensor): The vectors to rotate, of shape ``(..., 3)``.

    Returns:
        The rotated vectors, of the same shape as ``b``.

    """
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, 1:]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotates vectors by the inverse of quaternions.

    Args:
        q (torch.tensor): The quaternions, of shape ``(N, 4)`` and in ``[w, x, y, z]`` order.
        v (torch.tensor): The vectors to rotate, of shape ``(N, 3)``.

    Returns:
        The rotated vectors, of shape ``(N, 3)``.

    """
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c
