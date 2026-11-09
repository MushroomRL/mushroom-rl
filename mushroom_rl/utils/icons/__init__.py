from pathlib import Path

ICON_SIZES = (16, 32, 64, 256)


def icon_path(size):
    """
    Path of the MushroomRL window icon.

    Args:
        size (int): side of the square icon in pixels, one of ``ICON_SIZES``.

    Returns:
        The path of the PNG file of the icon.

    """
    assert size in ICON_SIZES, f'There is no {size}x{size} icon, the available sizes are {ICON_SIZES}.'

    return (Path(__file__).resolve().parent / f'mushroom_{size}.png').as_posix()
