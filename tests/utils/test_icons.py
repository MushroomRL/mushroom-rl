import pygame

from mushroom_rl.utils.icons import ICON_SIZES, icon_path


def test_icons():
    assert ICON_SIZES == (16, 32, 64, 256)

    for size in ICON_SIZES:
        assert pygame.image.load(icon_path(size)).get_size() == (size, size)
