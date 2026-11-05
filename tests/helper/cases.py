import pytest
import torch


def needs_cuda():
    return pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda not available')


def devices():
    return ['cpu', pytest.param('cuda', marks=needs_cuda())]


def backends():
    return ['numpy', 'torch', 'list']


def backend_devices(*names):
    names = names or ('numpy', 'torch', 'list')
    cases = list()
    for name in names:
        if name == 'torch':
            cases += [pytest.param('torch', 'cpu', id='torch-cpu'),
                      pytest.param('torch', 'cuda', id='torch-cuda', marks=needs_cuda())]
        else:
            cases.append(pytest.param(name, None, id=name))
    return cases
