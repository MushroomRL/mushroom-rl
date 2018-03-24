import errno
import os
from os.path import join as join_paths


def mk_dir_recursive(dir_path):
    """
    Create a directory and, if needed, all the directory tree. Differently from
    os.mkdir, this function does not raise exception when the directory already
    exists.

    Args:
        dir_path (str): the path of the directory to create.

    """
    if os.path.isdir(dir_path):
        return
    h, t = os.path.split(dir_path)  # head/tail
    if not os.path.isdir(h):
        mk_dir_recursive(h)

    new_path = join_paths(h, t)
    if not os.path.isdir(new_path):
        os.mkdir(new_path)


def force_symlink(src, dst):
    """
    Create a symlink deleting the previous one, if it already exists.

    Args:
        src (str): source;
        dst (str): destination.

    """
    src = src.rstrip('/')
    dst = dst.rstrip('/')

    try:
        os.symlink(src, dst)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(dst)
            os.symlink(src, dst)
