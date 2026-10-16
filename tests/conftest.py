import pytest
import torch

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
