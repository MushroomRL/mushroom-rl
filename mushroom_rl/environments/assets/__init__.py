from pathlib import Path


def asset_path(*parts):
    """
    Path of a file in the environment assets.

    Args:
        *parts (str): components of the file path relative to the assets folder, starting with the folder of the
            robot or scene, e.g. ``asset_path('go2', 'go2.usd')``.

    Returns:
        The path of the file.

    """
    return Path(__file__).resolve().parent.joinpath(*parts).as_posix()
