from . import _require_launched  # noqa: F401 -- raises if Isaac Sim hasn't been launched yet

import numpy as np

import omni.replicator.core as rep
from isaacsim.core.rendering_manager import RenderingManager
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf

from mushroom_rl.utils.viewer import ImageViewer


class IsaacViewer:
    """
    The camera looking at the simulated scene, and the window showing what it sees.

    Isaac Sim renders into a render product rather than into a window, so a frame is always available: the same
    image is displayed and, when the environment is recording, handed back to it.

    """

    def __init__(self, dt, camera_position=(5, 0, 4), camera_target=(0, 0, 0), render_product_size=(1280, 720)):
        """
        Constructor.

        Args:
            dt (float): The interval between two rendered frames.
            camera_position (tuple): Where the camera is placed, in world coordinates.
            camera_target (tuple): The point the camera looks at, in world coordinates.
            render_product_size (tuple): The (width, height) of the recorded and displayed frames.

        """
        RenderingManager.set_dt(dt)

        self._rp_size = render_product_size
        self._image_viewer = None

        viewport_api = get_viewport_from_window_name("Viewport")
        viewport_api.set_active_camera("/OmniverseKit_Persp")

        self._camera_state = ViewportCameraState("/OmniverseKit_Persp", viewport_api)
        self._camera_state.set_position_world(Gf.Vec3d(camera_position), True)
        self._camera_state.set_target_world(Gf.Vec3d(camera_target), True)

        rp = rep.create.render_product("/OmniverseKit_Persp", self._rp_size)
        self._rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb", do_array_copy=False, device="cuda")
        self._rgb_annot.attach(rp)

    def render(self):
        """
        Renders the scene and displays the frame.

        Returns:
            The rendered frame, as a (height, width, 3) array of bytes.

        """
        self.refresh()
        data = self._rgb_annot.get_data().numpy()[..., :3]

        if data.size == 0:
            data = np.zeros((self._rp_size[1], self._rp_size[0], 3), dtype=np.uint8)  # must be Height, Width, rgb

        if self._image_viewer is None:
            self._image_viewer = ImageViewer(self._rp_size, 0)
        self._image_viewer.display(data)

        return data

    def close(self):
        """
        Closes the window, if one is open. A later call to :meth:`render` opens it again.

        """
        if self._image_viewer is not None:
            self._image_viewer.close()
            self._image_viewer = None

    @staticmethod
    def refresh():
        """
        Renders the scene, keeping the Isaac Sim window up to date, without reading the frame back.

        """
        RenderingManager.render()

    @property
    def render_product_size(self):
        return self._rp_size
