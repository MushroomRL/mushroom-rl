from mushroom.utils.folder import *


def test_folder_utils(tmpdir):
    dir_path_1 = tmpdir / 'foo/bar'
    dir_path_2 = tmpdir / 'foo/foobar'
    mk_dir_recursive(str(dir_path_1))
    mk_dir_recursive(str(dir_path_2))
    filename = dir_path_1 / 'test.txt'
    filename.write("content")

    assert len(dir_path_2.listdir()) == 0
    assert len(dir_path_1.listdir()) == 1

    symlink = tmpdir / 'foo/foofoo'
    force_symlink(str(dir_path_2), str(symlink))
    force_symlink(str(dir_path_1), str(symlink))

    filename_linked = symlink / 'test.txt'
    assert filename_linked.read() == "content"





