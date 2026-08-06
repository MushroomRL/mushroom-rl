import time
from itertools import cycle

import glfw
import mujoco
import mujoco.viewer
import numpy as np


class MujocoViewer:
    """
    Class that creates a viewer for mujoco environments.

    Thin wrapper over mujoco.viewer.launch_passive (interactive window) and mujoco.Renderer (headless
    rendering and recording).

    """

    def __init__(self, model, dt, width=640, height=480, start_paused=False,
                 record=False, camera_params=None, default_camera_mode="static",
                 hide_menu_on_startup=True, hide_overlay_on_startup=False, geom_group_visualization_on_startup=None,
                 headless=False, reference_frame_visualization_on_startup=0):
        """
        Constructor.

        Args:
            model: Mujoco model.
            dt (float): Timestep of the environment, (not the simulation).
            width (int, 640): Width of the offscreen buffer used for headless rendering and for
                recording alongside an open window.
            height (int, 480): Height of the offscreen buffer.
            start_paused (bool, False): If True, the rendering is paused in the beginning of the simulation.
            record (bool, False): If true, frames are returned during rendering.
            camera_params (dict, None): Dictionary of dictionaries including custom parameterization of the three
                cameras. Checkout the function get_default_camera_params() to know what parameters are expected. If
                None, the defaults from get_default_camera_params() are used for all three cameras.
            default_camera_mode (str, "static"): Camera mode active on startup, one of "static", "follow",
                "top_static".
            hide_menu_on_startup (bool, True): If True, the native Simulate UI panels are hidden on startup.
            hide_overlay_on_startup (bool, False): If True, our own text overlay is hidden on startup.
            geom_group_visualization_on_startup (int/list, None): int or list defining which geom group_ids should
                be visualized on startup. If None, all are visualized.
            headless (bool, False): If True, render will be done in headless mode.
            reference_frame_visualization_on_startup (int, 0): Specifies the group of reference frames that should
                be visualized.

        """
        self.dt = dt

        self._headless = headless
        self._model = model
        self._hide_menu = hide_menu_on_startup
        self._hide_overlay = hide_overlay_on_startup

        self._width, self._height = self._grow_offscreen_buffer(model, width, height)

        self._scene_option = mujoco.MjvOption()
        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self._camera)
        if camera_params is None:
            self._camera_params = self.get_default_camera_params()
        else:
            self._camera_params = self._assert_camera_params(camera_params)
        all_camera_modes = ("static", "follow", "top_static")
        self._camera_mode_iter = cycle(all_camera_modes)
        self._camera_mode = None
        self._camera_mode_target = next(self._camera_mode_iter)
        assert default_camera_mode in all_camera_modes
        while self._camera_mode_target != default_camera_mode:
            self._camera_mode_target = next(self._camera_mode_iter)

        self._init_scene_option(self._scene_option, geom_group_visualization_on_startup,
                                reference_frame_visualization_on_startup)
        if headless:
            self._set_camera(self._camera)
        self._geom_group_visualization_on_startup = geom_group_visualization_on_startup
        self._reference_frame_visualization_on_startup = reference_frame_visualization_on_startup

        self._loop_count = 0
        self._time_per_render = 1 / 60.
        self._run_speed_factor = 1.0
        self._paused = start_paused

        self._handle = None
        self._renderer = None
        self._record_offscreen = headless or record
        if self._record_offscreen:
            self._renderer = mujoco.Renderer(self._model, height=self._height, width=self._width)

    def load_new_model(self, model):
        """
        Loads a new model to the viewer, and resets the scene and context.
        This is used in MultiMujoco environments.

        Args:
            model: Mujoco model.

        """
        self._model = model
        self._width, self._height = self._grow_offscreen_buffer(model, self._width, self._height)

        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._record_offscreen:
            self._renderer = mujoco.Renderer(self._model, height=self._height, width=self._width)

        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def render(self, data, record):
        """
        Main rendering function.

        Args:
            data: Mujoco data structure.
            record (bool): If true, frames are returned during rendering.

        Returns:
            If record is True, frames are returned during rendering, else None.

        """
        if not self._headless and self._handle is None:
            self._launch_passive(data)

        if self._paused:
            while self._paused:
                frame = self._render_frame(data, record)

        if record:
            self._loop_count = 1
        else:
            self._loop_count += self.dt / (self._time_per_render * self._run_speed_factor)
        while self._loop_count > 0:
            frame = self._render_frame(data, record)
            self._loop_count -= 1

        if record:
            return frame

    def stop(self):
        """
        Closes the viewer and releases the renderer.

        """
        if self._handle is not None:
            self._handle.close()
            while self._handle.is_running():
                time.sleep(0.01)
            time.sleep(0.1)
            self._handle = None
        if self._renderer is not None:
            self._renderer.close()

    def _key_callback(self, keycode):
        """
        Key callback registered with mujoco.viewer.launch_passive.

        Args:
            keycode (int): glfw key code of the pressed key.

        """
        if keycode == glfw.KEY_SPACE:
            self._paused = not self._paused

        if keycode in (glfw.KEY_LEFT_ALT, glfw.KEY_RIGHT_ALT):
            self._camera_mode_target = next(self._camera_mode_iter)

        if keycode == glfw.KEY_S:
            self._run_speed_factor = max(self._run_speed_factor / 2.0, 1 / 1024)

        if keycode == glfw.KEY_F:
            self._run_speed_factor = min(self._run_speed_factor * 2.0, 1024)

    def _create_overlay(self):
        """
        Refreshes the small top-right text overlay.

        """
        self._handle.set_texts([
            (None, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
             "TAB\nSPACE\nALT\n[ / ]",
             "MuJoCo controls\nPause\nCycle camera (%s)\nCycle camera (if defined by the model)" % self._camera_mode),
            (None, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
             "\nS / F", "\nRun speed = %.3fx" % self._run_speed_factor)])

    def _launch_passive(self, data):
        """
        Lazily launches the interactive window on the first render() call.

        Args:
            data: Mujoco data structure.

        """
        self._handle = mujoco.viewer.launch_passive(
            self._model, data, key_callback=self._key_callback,
            show_left_ui=not self._hide_menu, show_right_ui=not self._hide_menu)
        self._init_scene_option(self._handle.opt, self._geom_group_visualization_on_startup,
                                self._reference_frame_visualization_on_startup)
        self._camera_mode = None
        self._set_camera(self._handle.cam)

    def _render_frame(self, data, record):
        """
        Renders a single frame (headless or windowed) and updates the pacing statistics.

        Args:
            data: Mujoco data structure.
            record (bool): If true, the rendered frame is returned.

        Returns:
            The rendered frame if record is True, else None.

        """
        render_start = time.time()

        frame = self._render_headless(data) if self._headless else self._render_windowed(data, record)

        measured = time.time() - render_start
        self._time_per_render = max(0.9 * self._time_per_render + 0.1 * measured, 1 / 240)

        return frame

    def _render_headless(self, data):
        """
        Renders one offscreen frame, using the camera pose set once at construction.

        Args:
            data: Mujoco data structure.

        Returns:
            The rendered frame.

        """
        self._renderer.update_scene(data, camera=self._camera, scene_option=self._scene_option)

        return self._renderer.render()

    def _render_windowed(self, data, record):
        """
        Refreshes the interactive window and, if requested, renders an offscreen frame alongside it
        using the same camera/scene_option the window is currently showing.

        Args:
            data: Mujoco data structure.
            record (bool): If true, an offscreen frame is rendered and returned.

        Returns:
            The rendered frame if record is True, else None.

        """
        if not self._hide_overlay:
            self._create_overlay()
        self._handle.sync()
        if not self._handle.is_running():
            self.stop()
            exit(0)

        frame = None
        if record:
            self._renderer.update_scene(data, camera=self._handle.cam, scene_option=self._handle.opt)
            frame = self._renderer.render()

        return frame

    def _set_camera(self, camera):
        """
        Sets the given camera to the current camera mode target. Allowed camera modes are "follow"
        in which the model is tracked, "static" that is a static camera at the default camera
        position, and "top_static" that is a static camera on top of the model.

        Args:
            camera: MjvCamera (offscreen rendering) or the live mujoco.viewer.Handle.cam (interactive
                window) to mutate.

        """
        if self._camera_mode_target == "follow":
            if self._camera_mode != "follow":
                camera.fixedcamid = -1
                camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                camera.trackbodyid = 0
                self._set_camera_properties(camera, self._camera_mode_target)
        elif self._camera_mode_target == "static":
            if self._camera_mode != "static":
                camera.fixedcamid = 0
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                camera.trackbodyid = -1
                self._set_camera_properties(camera, self._camera_mode_target)
        elif self._camera_mode_target == "top_static":
            if self._camera_mode != "top_static":
                camera.fixedcamid = 0
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                camera.trackbodyid = -1
                self._set_camera_properties(camera, self._camera_mode_target)

    def _set_camera_properties(self, camera, mode):
        """
        Sets the "distance", "elevation", and "azimuth" properties of the given camera, as well as
        the camera mode, based on the provided mode.

        Args:
            camera: MjvCamera or mujoco.viewer.Handle.cam to mutate.
            mode (str): Camera mode. (either "follow", "static", or "top_static")

        """
        cam_params = self._camera_params[mode]
        camera.distance = cam_params["distance"]
        camera.elevation = cam_params["elevation"]
        camera.azimuth = cam_params["azimuth"]
        if "lookat" in cam_params:
            camera.lookat = np.array(cam_params["lookat"])
        self._camera_mode = mode

    def _assert_camera_params(self, camera_params):
        """
        Asserts if the provided camera parameters are valid or not. Also, if
        properties of some camera types are not specified, the default parameters
        are used.

        Args:
            camera_params (dict): Dictionary of dictionaries containig parameters for each camera type.

        Returns:
            Dictionary of dictionaries with parameters for each camera type.

        """
        default_camera_params = self.get_default_camera_params()

        # check if the provided camera types and parameters are valid
        for cam_type in camera_params.keys():
            assert cam_type in default_camera_params.keys(), f"Camera type \"{cam_type}\" is unknown. Allowed " \
                                                             f"camera types are {list(default_camera_params.keys())}."
            for param in camera_params[cam_type].keys():
                assert param in default_camera_params[cam_type].keys(), \
                    (f"Parameter \"{param}\" of camera type \"{cam_type}\" is unknown. "
                     f"Allowed parameters are {list(default_camera_params[cam_type].keys())}")

        # add default parameters if not specified
        for cam_type in default_camera_params.keys():
            if cam_type not in camera_params.keys():
                camera_params[cam_type] = default_camera_params[cam_type]
            else:
                for param in default_camera_params[cam_type].keys():
                    if param not in camera_params[cam_type].keys():
                        camera_params[cam_type][param] = default_camera_params[cam_type][param]

        return camera_params

    @staticmethod
    def get_default_camera_params():
        """
        Getter for default camera paramterization.

        Returns:
            Dictionary of dictionaries with default parameters for each camera type.

        """
        return dict(static=dict(distance=15.0, elevation=-45.0, azimuth=90.0, lookat=np.array([0.0, 0.0, 0.0])),
                    follow=dict(distance=3.5, elevation=0.0, azimuth=90.0),
                    top_static=dict(distance=5.0, elevation=-90.0, azimuth=90.0, lookat=np.array([0.0, 0.0, 0.0])))

    @staticmethod
    def _grow_offscreen_buffer(model, width, height):
        """
        Grows the model's offscreen buffer (mujoco.Renderer refuses to exceed it) to fit the
        requested size, if needed. Must run before any mujoco.Renderer is constructed for this model.

        Args:
            model: Mujoco model whose vis.global_ offwidth/offheight may need growing.
            width (int): Requested width.
            height (int): Requested height.

        Returns:
            The (possibly grown) (width, height) that the offscreen buffer now supports.

        """
        if width > model.vis.global_.offwidth or height > model.vis.global_.offheight:
            width = max(width, model.vis.global_.offwidth)
            height = max(height, model.vis.global_.offheight)
            model.vis.global_.offwidth = width
            model.vis.global_.offheight = height

        return width, height

    @staticmethod
    def _init_scene_option(scene_option, geom_group_visualization_on_startup,
                           reference_frame_visualization_on_startup):
        """
        Applies the startup geom-group and reference-frame visualization settings to the given
        MjvOption (either the owned offscreen one, or the live mujoco.viewer.Handle.opt).

        Args:
            scene_option: MjvOption to mutate.
            geom_group_visualization_on_startup (int/list): See the constructor.
            reference_frame_visualization_on_startup (int): See the constructor.

        """
        if geom_group_visualization_on_startup is not None:
            assert isinstance(geom_group_visualization_on_startup, (list, int))
            if type(geom_group_visualization_on_startup) is not list:
                geom_group_visualization_on_startup = [geom_group_visualization_on_startup]
            for group_id, _ in enumerate(scene_option.geomgroup):
                if group_id not in geom_group_visualization_on_startup:
                    scene_option.geomgroup[group_id] = False

        scene_option.frame = reference_frame_visualization_on_startup
