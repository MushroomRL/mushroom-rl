import pygame
import numpy as np

from mushroom_rl.utils.viewer import Viewer
from mushroom_rl.utils.viewer.pygame_viewer import ImageViewer


def test_window():
    square = Viewer(10, 10, min_scale=40)
    corridor = Viewer(2, 20, min_scale=40)
    wide = Viewer(48, 1, min_scale=40)

    assert square.size == (500, 500)
    assert corridor.size == (500, 1080)
    assert wide.size == (1920, 100)

    for viewer in (square, corridor, wide):
        assert viewer.fits
        assert viewer.size[0] >= 500 and viewer.size[1] >= 100

    assert not Viewer(60, 60, min_scale=40).fits
    assert not Viewer(100, 1, min_scale=40).fits
    assert not Viewer(48, 2084, min_scale=40).fits

    padded = Viewer(48, 1, min_scale=40)
    exact = Viewer(10, 10, min_scale=40)

    assert np.allclose(exact._margin, [0., 0.])
    assert np.allclose(padded._margin, [0., .75])


def test_background_image():
    image = np.full((64, 64, 3), 255.)

    padded = Viewer(2, 20, min_scale=40)
    padded.background_image(image)
    padded_columns = np.argwhere(padded.get_frame().any(-1))[:, 1]
    padded.close()

    exact = Viewer(10, 10, min_scale=40)
    exact.background_image(image)
    exact_columns = np.argwhere(exact.get_frame().any(-1))[:, 1]
    exact.close()

    assert padded_columns.min() == 196 and padded_columns.max() == 303
    assert exact_columns.min() == 0 and exact_columns.max() == 499


def test_grid_borders():
    viewer = Viewer(5, 5, min_scale=40)
    viewer.grid(5, 5)
    frame = viewer.get_frame()
    viewer.close()

    rows = [row for row in range(frame.shape[0]) if frame[row].all()]
    columns = [column for column in range(frame.shape[1]) if frame[:, column].all()]

    assert rows == [0, 100, 200, 300, 400, 499]
    assert columns == [0, 100, 200, 300, 400, 499]


def test_image_viewer():
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    image[5, 7] = [10, 20, 30]
    viewer = ImageViewer((40, 30), 0.)
    viewer.display(np.full((30, 40, 3), 255, dtype=np.uint8))
    viewer.display(image)

    assert viewer.size == (40, 30)
    assert np.array_equal(pygame.surfarray.array3d(pygame.display.get_surface()), image.transpose(1, 0, 2))

    viewer.close()
    ImageViewer((40, 30), 0., headless=True).display(image)

    assert not pygame.display.get_init()
