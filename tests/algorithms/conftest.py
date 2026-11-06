import pytest


@pytest.fixture(autouse=True)
def reset_default_device(request):
    from mushroom_rl.utils import TorchUtils

    TorchUtils.set_default_device(request.getfixturevalue('device') if 'device' in request.fixturenames else 'cpu')
    yield
    TorchUtils.set_default_device('cpu')
