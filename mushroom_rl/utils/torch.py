import torch
import numpy as np


def set_weights(parameters, weights, use_cuda):
    """
    Function used to set the value of a set of torch parameters given a
    vector of values.

    Args:
        parameters (list): list of parameters to be considered;
        weights (numpy.ndarray): array of the new values for
            the parameters;
        use_cuda (bool): whether the parameters are cuda tensors or not;

    """
    idx = 0
    for p in parameters:
        shape = p.data.shape

        c = 1
        for s in shape:
            c *= s

        w = np.reshape(weights[idx:idx + c], shape)

        if not use_cuda:
            w_tensor = torch.from_numpy(w).type(p.data.dtype)
        else:
            w_tensor = torch.from_numpy(w).type(p.data.dtype).cuda()

        p.data = w_tensor
        idx += c

    assert idx == weights.size


def get_weights(parameters):
    """
    Function used to get the value of a set of torch parameters as
    a single vector of values.

    Args:
        parameters (list): list of parameters to be considered.

    Returns:
        A numpy vector consisting of all the values of the vectors.

    """
    weights = list()

    for p in parameters:
        w = p.data.detach().cpu().numpy()
        weights.append(w.flatten())

    weights = np.concatenate(weights, 0)

    return weights


def zero_grad(parameters):
    """
    Function used to set to zero the value of the gradient of a set
    of torch parameters.

    Args:
        parameters (list): list of parameters to be considered.

    """

    for p in parameters:
        if p.grad is not None:
           p.grad.detach_()
           p.grad.zero_()


def get_gradient(params):
    """
    Function used to get the value of the gradient of a set of
    torch parameters.

    Args:
        parameters (list): list of parameters to be considered.

    """
    views = []
    for p in params:
        if p.grad is None:
            view = p.new(p.numel()).zero_()
        else:
            view = p.grad.view(-1)
        views.append(view)
    return torch.cat(views, 0)


def to_float_tensor(x, use_cuda=False):
    """
    Function used to convert a numpy array to a float torch tensor.

    Args:
        x (np.ndarray): numpy array to be converted as torch tensor;
        use_cuda (bool): whether to build a cuda tensors or not.

    Returns:
        A float tensor build from the values contained in the input array.

    """
    x = torch.tensor(x, dtype=torch.float)
    return x.cuda() if use_cuda else x


def to_int_tensor(x, use_cuda=False):
    """
    Function used to convert a numpy array to a float torch tensor.

    Args:
        x (np.ndarray): numpy array to be converted as torch tensor;
        use_cuda (bool): whether to build a cuda tensors or not.

    Returns:
        A float tensor build from the values contained in the input array.

    """
    x = torch.tensor(x, dtype=torch.int)
    return x.cuda() if use_cuda else x


def update_optimizer_parameters(optimizer, new_parameters):
    if len(optimizer.state) > 0:
        for p_old, p_new in zip(optimizer.param_groups[0]['params'], new_parameters):
            data = optimizer.state[p_old]
            del optimizer.state[p_old]
            optimizer.state[p_new] = data

    optimizer.param_groups[0]['params'] = new_parameters


class CategoricalWrapper(torch.distributions.Categorical):
    """
    Wrapper for the Torch Categorical distribution.

    Needed to convert a vector of mushroom discrete action in an input with the proper shape of the original
    distribution implemented in torch

    """
    def __init__(self, logits):
        super().__init__(logits=logits)

    def log_prob(self, value):
        return super().log_prob(value.squeeze())


class DiagonalMultivariateGaussian(torch.distributions.Normal):
    """
    Wrapper for the Torch Normal distribution, implementing a diagonal distribution.

    It behaves as the MultivariateNormal distribution, but avoids the computation of the full covariance matrix,
    optimizing the computation time, particulalrly when a high dimensional vector is sampled.

    """
    def __init__(self, loc, scale):
        super().__init__(loc=loc, scale=scale)

    def log_prob(self, value):
        return torch.sum(super().log_prob(value), -1)