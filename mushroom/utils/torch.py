import torch
import numpy as np


def set_weights(parameters, weights, use_cuda):
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
    weights = list()

    for p in parameters:
        w = p.data.detach().cpu().numpy()
        weights.append(w.flatten())

    weights = np.concatenate(weights, 0)

    return weights


def zero_grad(parameters):
    for p in parameters:
        if p.grad is not None:
           p.grad.detach_()
           p.grad.zero_()


def get_gradient(params):
    views = []
    for p in params:
        if p.grad is None:
            view = p.new(p.numel()).zero_()
        else:
            view = p.grad.view(-1)
        views.append(view)
    return torch.cat(views, 0)


def to_float_tensor(x, use_cuda=False):
    x = torch.tensor(x, dtype=torch.float)
    return x.cuda() if use_cuda else x
