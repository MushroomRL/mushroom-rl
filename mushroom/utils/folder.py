import errno
import os
from os.path import join as join_paths


def mk_dir_recursive(dir_path):

    if os.path.isdir(dir_path):
        return
    h, t = os.path.split(dir_path)  # head/tail
    if not os.path.isdir(h):
        mk_dir_recursive(h)

    new_path = join_paths(h, t)
    if not os.path.isdir(new_path):
        os.mkdir(new_path)


def force_symlink(file1, file2):
    file1 = file1.rstrip('/')
    file2 = file2.rstrip('/')

    try:
        os.symlink(file1, file2)
    except OSError, e:
        print e.message
        if e.errno == errno.EEXIST:
            os.remove(file2)
            os.symlink(file1, file2)
