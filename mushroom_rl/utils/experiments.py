from pathlib import Path


def get_root_dir(script_file, root_name='examples'):
    """
    Return the closest ancestor of the given script named ``root_name``. As the lookup starts from the
    script location, the result does not depend on the working directory, nor on how deep the script is
    stored inside the root directory.

    Args:
        script_file (str, Path): path of the script, typically its ``__file__``;
        root_name (str, 'examples'): name of the root directory to look for.

    Returns:
        The path of the root directory.

    """
    path = Path(script_file).resolve()

    for directory in path.parents:
        if directory.name == root_name:
            return directory

    raise ValueError(f'No ancestor directory named {root_name} for {path}')


def get_log_dir(script_file, root_name='examples'):
    """
    Return the logging directory shared by every script of an experiments folder, i.e. the ``logs``
    subdirectory of the root directory.

    Args:
        script_file (str, Path): path of the script, typically its ``__file__``;
        root_name (str, 'examples'): name of the root directory to look for.

    Returns:
        The path of the logging directory.

    """
    return get_root_dir(script_file, root_name) / 'logs'


def get_data_dir(script_file, root_name='examples'):
    """
    Return the data directory shared by every script of an experiments folder, i.e. the ``data``
    subdirectory of the root directory.

    Args:
        script_file (str, Path): path of the script, typically its ``__file__``;
        root_name (str, 'examples'): name of the root directory to look for.

    Returns:
        The path of the data directory.

    """
    return get_root_dir(script_file, root_name) / 'data'


def select_class(name, classes):
    """
    Select a class by name among a set of accepted ones. This is meant to turn a command line argument
    into the class to be used, letting every script state which classes it accepts.

    Args:
        name (str): name of the class to select;
        classes (list): the accepted classes.

    Returns:
        The class in ``classes`` whose name is ``name``.

    """
    for cls in classes:
        if cls.name() == name:
            return cls

    accepted = ', '.join(cls.name() for cls in classes)
    raise ValueError(f'Unknown class {name}. Accepted classes are: {accepted}')
