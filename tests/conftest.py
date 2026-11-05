import os

import pytest

# Pin the MKL and ATen kernel paths before torch loads, so that every CPU computes the same numbers
os.environ['MKL_CBWR'] = 'COMPATIBLE'
os.environ['ATEN_CPU_CAPABILITY'] = 'avx2'
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import torch  # noqa: E402

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def pytest_addoption(parser):
    parser.addoption("--isaac", action="store_true", help="skip the tests broken by the packages Isaac Sim pins")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--isaac"):
        # Isaac Sim pins mujoco / mujoco-warp versions older than the ones these tests were recorded with
        pinned_paths = (
            "tests/environments/mujoco_envs/",
            "tests/environments/mujoco_warp_envs/",
            "tests/environments/test_mujoco_warp.py",
        )
        skip = pytest.mark.skip(reason="needs package versions newer than the ones pinned by Isaac Sim")
        for item in items:
            if item.nodeid.startswith(pinned_paths):
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def reset_default_device():
    from mushroom_rl.utils import TorchUtils

    TorchUtils.set_default_device('cpu')
    yield
    TorchUtils.set_default_device('cpu')
