import os
import glfw
import mujoco
import time
import collections
from itertools import cycle
import numpy as np


def _import_egl(width, height):
    from mujoco.egl import GLContext

    return GLContext(width, height)


def _import_glfw(width, height):
    from mujoco.glfw import GLContext

    return GLContext(width, height)


def _import_osmesa(width, height):
    from mujoco.osmesa import GLContext

    return GLContext(width, height)


_ALL_RENDERERS = collections.OrderedDict(
    [
        ("glfw", _import_glfw),
        ("egl", _import_egl),
        ("osmesa", _import_osmesa),
    ]
)


class MujocoViewer:
    """
    Class that creates a viewer for mujoco environments.

    """

    def __init__(self, model, dt, width=1920, height=1080, start_paused=False,
                 custom_render_callback=None, record=False, camera_params=None,
                 default_camera_mode="static", hide_menu_on_startup=None,
                 geom_group_visualization_on_startup=None, headless=False):
        """
        Constructor.

        Args:
            model: Mujoco model.
            dt (float): Timestep of the environment, (not the simulation).
            width (int): Width of the viewer window.
            height (int): Height of the viewer window.
            start_paused (bool): If True, the rendering is paused in the beginning of the simulation.
            custom_render_callback (func): Custom render callback function, which is supposed to be called
                during rendering.
            record (bool): If true, frames are returned during rendering.
            camera_params (dict): Dictionary of dictionaries including custom parameterization of the three cameras.
                Checkout the function get_default_camera_params() to know what parameters are expected. Is some camera
                type specification or parameter is missing, the default one is used.
            hide_menu_on_startup (bool): If True, the menu is hidden on startup.
            geom_group_visualization_on_startup (int/list): int or list defining which geom group_ids should be
                visualized on startup. If None, all are visualized.
            headless (bool): If True, render will be done in headless mode.

        """

        if hide_menu_on_startup is None and headless:
            hide_menu_on_startup = True
        elif hide_menu_on_startup is None and not headless:
            hide_menu_on_startup = False

        self.button_left = False
        self.button_right = False
        self.button_middle = False
        self.last_x = 0
        self.last_y = 0
        self.dt = dt

        self.frames = 0
        self.start_time = time.time()

        self._headless = headless
        self._model = model
        self._font_scale = 100
        
        if headless:
            # use the OpenGL render that is available on the machine
            self._opengl_context = self.setup_opengl_backend_headless(width, height)
            self._opengl_context.make_current()
            self._width, self._height = self.update_headless_size(width, height)
        else:
            # use glfw
            self._width, self._height = width, height
            glfw.init()
            glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, 0)
            self._window = glfw.create_window(width=self._width, height=self._height,
                                              title="MuJoCo", monitor=None, share=None)
            glfw.make_context_current(self._window)
            glfw.set_mouse_button_callback(self._window, self.mouse_button)
            glfw.set_cursor_pos_callback(self._window, self.mouse_move)
            glfw.set_key_callback(self._window, self.keyboard)
            glfw.set_scroll_callback(self._window, self.scroll)

        self._set_mujoco_buffers()
        
        if record and not headless:
            # dont allow to change the window size to have equal frame size during recording
            glfw.window_hint(glfw.RESIZABLE, False)

        self._viewport = mujoco.MjrRect(0, 0, self._width, self._height)
        self._loop_count = 0
        self._time_per_render = 1 / 60.
        self._run_speed_factor = 1.0
        self._paused = start_paused

        # Disable v_sync, so swap_buffers does not block
        # glfw.swap_interval(0)

        self._scene = mujoco.MjvScene(self._model, 1000)
        self._scene_option = mujoco.MjvOption()
        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self._camera)
        if camera_params is None:
            self._camera_params = self.get_default_camera_params()
        else:
            self._camera_params = self._assert_camera_params(camera_params)
        self._all_camera_modes = ("static", "follow", "top_static")
        self._camera_mode_iter = cycle(self._all_camera_modes)
        self._camera_mode = None
        self._camera_mode_target = next(self._camera_mode_iter)
        assert default_camera_mode in self._all_camera_modes
        while self._camera_mode_target != default_camera_mode:
            self._camera_mode_target = next(self._camera_mode_iter)
        self._set_camera()

        self.custom_render_callback = custom_render_callback

        self._overlay = {}
        self._hide_menu = hide_menu_on_startup

        if geom_group_visualization_on_startup is not None:
            assert type(geom_group_visualization_on_startup) == list or type(geom_group_visualization_on_startup) == int
            if type(geom_group_visualization_on_startup) is not list:
                geom_group_visualization_on_startup = [geom_group_visualization_on_startup]
            for group_id, _ in enumerate(self._scene_option.geomgroup):
                if group_id not in geom_group_visualization_on_startup:
                    self._scene_option.geomgroup[group_id] = False

    def load_new_model(self, model):
        """
        Loads a new model to the viewer, and resets the scene and context.
        This is used in MultiMujoco environments.

        Args:
            model: Mujoco model.

        """

        self._model = model
        self._scene = mujoco.MjvScene(model, 1000)
        self._context = mujoco.MjrContext(model, mujoco.mjtFontScale(self._font_scale))


    def mouse_button(self, window, button, act, mods):
        """
        Mouse button callback for glfw.

        Args:
            window: glfw window.
            button: glfw button id.
            act: glfw action.
            mods: glfw mods.

        """

        self.button_left = glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.button_right = glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        self.button_middle = glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS

        self.last_x, self.last_y = glfw.get_cursor_pos(self._window)

    def mouse_move(self, window, x_pos, y_pos):
        """
        Mouse mode callback for glfw.

        Args:
            window:  glfw window.
            x_pos: Current mouse x position.
            y_pos: Current mouse y position.

        """

        if not self.button_left and not self.button_right and not self.button_middle:
            return

        dx = x_pos - self.last_x
        dy = y_pos - self.last_y
        self.last_x = x_pos
        self.last_y = y_pos

        width, height = glfw.get_window_size(self._window)

        mod_shift = glfw.get_key(self._window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(self._window,
                                                                                                  glfw.KEY_RIGHT_SHIFT) == glfw.PRESS

        if self.button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        mujoco.mjv_moveCamera(self._model, action, dx / width, dy / height, self._scene, self._camera)

    def keyboard(self, window, key, scancode, act, mods):
        """
        Keyboard callback for glfw.

        Args:
            window: glfw window.
            key: glfw key event.
            scancode: glfw scancode.
            act: glfw action.
            mods: glfw mods.

        """

        if act != glfw.RELEASE:
            return

        if key == glfw.KEY_SPACE:
            self._paused = not self._paused

        if key == glfw.KEY_C:
            self._scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = not self._scene_option.flags[
                mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]
            self._scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = not self._scene_option.flags[
                mujoco.mjtVisFlag.mjVIS_CONSTRAINT]

        if key == glfw.KEY_T:
            self._scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = not self._scene_option.flags[
                mujoco.mjtVisFlag.mjVIS_TRANSPARENT]

        if key == glfw.KEY_0:
            self._scene_option.geomgroup[0] = not self._scene_option.geomgroup[0]

        if key == glfw.KEY_1:
            self._scene_option.geomgroup[1] = not self._scene_option.geomgroup[1]

        if key == glfw.KEY_2:
            self._scene_option.geomgroup[2] = not self._scene_option.geomgroup[2]

        if key == glfw.KEY_3:
            self._scene_option.geomgroup[3] = not self._scene_option.geomgroup[3]

        if key == glfw.KEY_4:
            self._scene_option.geomgroup[4] = not self._scene_option.geomgroup[4]

        if key == glfw.KEY_5:
            self._scene_option.geomgroup[5] = not self._scene_option.geomgroup[5]

        if key == glfw.KEY_6:
            self._scene_option.geomgroup[6] = not self._scene_option.geomgroup[6]

        if key == glfw.KEY_7:
            self._scene_option.geomgroup[7] = not self._scene_option.geomgroup[7]

        if key == glfw.KEY_8:
            self._scene_option.geomgroup[8] = not self._scene_option.geomgroup[8]

        if key == glfw.KEY_9:
            self._scene_option.geomgroup[9] = not self._scene_option.geomgroup[9]

        if key == glfw.KEY_TAB:
            self._camera_mode_target = next(self._camera_mode_iter)

        if key == glfw.KEY_S:
            self._run_speed_factor /= 2.0

        if key == glfw.KEY_F:
            self._run_speed_factor *= 2.0

        if key == glfw.KEY_E:
            self._scene_option.frame = not self._scene_option.frame

        if key == glfw.KEY_H:
            if self._hide_menu:
                self._hide_menu = False
            else:
                self._hide_menu = True

    def scroll(self, window, x_offset, y_offset):
        """
        Scrolling callback for glfw.

        Args:
            window: glfw window.
            x_offset: x scrolling offset.
            y_offset: y scrolling offset.

        """

        mujoco.mjv_moveCamera(self._model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, 0.05 * y_offset, self._scene, self._camera)

    def _set_mujoco_buffers(self):
        self._context = mujoco.MjrContext(self._model, mujoco.mjtFontScale(self._font_scale))
        if self._headless:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._context)
            if self._context.currentBuffer != mujoco.mjtFramebuffer.mjFB_OFFSCREEN:
                raise RuntimeError("Offscreen rendering not supported")
        else:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self._context)
            if self._context.currentBuffer != mujoco.mjtFramebuffer.mjFB_WINDOW:
                raise RuntimeError("Window rendering not supported")

    def update_headless_size(self, width, height):
        _context = mujoco.MjrContext(self._model, mujoco.mjtFontScale(self._font_scale))
        if width > _context.offWidth or height > _context.offHeight:
            width = max(width, self._model.vis.global_.offwidth)
            height = max(height, self._model.vis.global_.offheight)
            
            if width != _context.offWidth or height != _context.offHeight:
                self._model.vis.global_.offwidth = width
                self._model.vis.global_.offheight = height

        return width, height

    def render(self, data, record):
        """
        Main rendering function.

        Args:
            data: Mujoco data structure.
            record (bool): If true, frames are returned during rendering.

        Returns:
            If record is True, frames are returned during rendering, else None.

        """

        def render_inner_loop(self):

            if not self._headless:
                self._create_overlay()

            render_start = time.time()

            mujoco.mjv_updateScene(self._model, data, self._scene_option, None, self._camera,
                                   mujoco.mjtCatBit.mjCAT_ALL,
                                   self._scene)

            if not self._headless:
                self._viewport.width, self._viewport.height = glfw.get_window_size(self._window)

            mujoco.mjr_render(self._viewport, self._scene, self._context)

            for gridpos, [t1, t2] in self._overlay.items():

                if self._hide_menu:
                    continue

                mujoco.mjr_overlay(
                    mujoco.mjtFont.mjFONT_SHADOW,
                    gridpos,
                    self._viewport,
                    t1,
                    t2,
                    self._context)

            if self.custom_render_callback is not None:
                self.custom_render_callback(self._viewport, self._context)

            if not self._headless:
                glfw.swap_buffers(self._window)
                glfw.poll_events()
                if glfw.window_should_close(self._window):
                    self.stop()
                    exit(0)

            self.frames += 1
            self._overlay.clear()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (time.time() - render_start)

        if self._paused:
            while self._paused:
                render_inner_loop(self)

        if record:
            self._loop_count = 1
        else:
            self._loop_count += self.dt / (self._time_per_render * self._run_speed_factor)
        while self._loop_count > 0:
            render_inner_loop(self)
            self._set_camera()
            self._loop_count -= 1

        if record:
            return self.read_pixels()

    def read_pixels(self, depth=False):
        """
        Reads the pixels from the glfw viewer.

        Args:
            depth (bool): If True, depth map is also returned.

        Returns:
            If depth is True, tuple of np.arrays (rgb and depth), else just a single
            np.array for the rgb image.

        """

        if self._headless:
            shape = (self._width, self._height)
        else:
            shape = glfw.get_framebuffer_size(self._window)

        if depth:
            rgb_img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
            depth_img = np.zeros((shape[1], shape[0], 1), dtype=np.float32)
            mujoco.mjr_readPixels(rgb_img, depth_img, self._viewport, self._context)
            return (np.flipud(rgb_img), np.flipud(depth_img))
        else:
            img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
            mujoco.mjr_readPixels(img, None, self._viewport, self._context)
            return np.flipud(img)

    def stop(self):
        """
        Destroys the glfw image.

        """
        if not self._headless:
            glfw.destroy_window(self._window)

    def _create_overlay(self):
        """
        This function creates and adds all overlays used in the viewer.

        """

        topleft = mujoco.mjtGridPos.mjGRID_TOPLEFT
        topright = mujoco.mjtGridPos.mjGRID_TOPRIGHT
        bottomleft = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
        bottomright = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT

        def add_overlay(gridpos, text1, text2="", make_new_line=True):
            if gridpos not in self._overlay:
                self._overlay[gridpos] = ["", ""]
            if make_new_line:
                self._overlay[gridpos][0] += text1 + "\n"
                self._overlay[gridpos][1] += text2 + "\n"
            else:
                self._overlay[gridpos][0] += text1
                self._overlay[gridpos][1] += text2

        add_overlay(
            bottomright,
            "Framerate:",
            str(int(1/self._time_per_render * self._run_speed_factor)), make_new_line=False)

        add_overlay(
            topleft,
            "Press SPACE to pause.")

        add_overlay(
            topleft,
            "Press H to hide the menu.")

        add_overlay(
            topleft,
            "Press TAB to switch cameras.")

        add_overlay(
            topleft,
            "Press T to make the model transparent.")

        add_overlay(
            topleft,
            "Press E to toggle reference frames.")

        add_overlay(
            topleft,
            "Press 0-9 to disable/enable geom group visualization.")

        visualize_contact = "On" if self._scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] else "Off"
        add_overlay(
            topleft,
            "Contact force visualization (Press C):", visualize_contact)

        add_overlay(
            topleft,
            "Camera mode:",
            self._camera_mode)
        
        add_overlay(
            topleft,
            "Run speed = %.3f x real time" %
            self._run_speed_factor,
            "[S]lower, [F]aster", make_new_line=False)

    def _set_camera(self):
        """
        Sets the camera mode to the current camera mode target. Allowed camera
        modes are "follow" in which the model is tracked, "static" that is a static
        camera at the default camera positon, and "top_static" that is a static
        camera on top of the model.

        """

        if self._camera_mode_target == "follow":
            if self._camera_mode != "follow":
                self._camera.fixedcamid = -1
                self._camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self._camera.trackbodyid = 0
                self._set_camera_properties(self._camera_mode_target)
        elif self._camera_mode_target == "static":
            if self._camera_mode != "static":
                self._camera.fixedcamid = 0
                self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                self._camera.trackbodyid = -1
                self._set_camera_properties(self._camera_mode_target)
        elif self._camera_mode_target == "top_static":
            if self._camera_mode != "top_static":
                self._camera.fixedcamid = 0
                self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                self._camera.trackbodyid = -1
                self._set_camera_properties(self._camera_mode_target)

    def _set_camera_properties(self, mode):
        """
        Sets the camera properties "distance", "elevation", and "azimuth"
        as well as the camera mode based on the provided mode.

        Args:
            mode (str): Camera mode. (either "follow", "static", or "top_static")

        """

        cam_params = self._camera_params[mode]
        self._camera.distance = cam_params["distance"]
        self._camera.elevation = cam_params["elevation"]
        self._camera.azimuth = cam_params["azimuth"]
        if "lookat" in cam_params:
            self._camera.lookat = np.array(cam_params["lookat"])
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
                assert param in default_camera_params[cam_type].keys(), f"Parameter \"{param}\" of camera type " \
                                                                        f"\"{cam_type}\" is unknown. Allowed " \
                                                                        f"parameters are" \
                                                                        f" {list(default_camera_params[cam_type].keys())}"

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


    def setup_opengl_backend_headless(self, width, height):

        backend = os.environ.get("MUJOCO_GL")
        if backend is not None:
            try:
                opengl_context = _ALL_RENDERERS[backend](width, height)
            except KeyError:
                raise RuntimeError(
                    "Environment variable {} must be one of {!r}: got {!r}.".format(
                        "MUJOCO_GL", _ALL_RENDERERS.keys(), backend
                    )
                )

        else:
            # iterate through all OpenGL backends to see which one is available
            for name, _ in _ALL_RENDERERS.items():
                try:
                    opengl_context = _ALL_RENDERERS[name](width, height)
                    backend = name
                    break
                except:  # noqa:E722
                    pass
            if backend is None:
                raise RuntimeError(
                    "No OpenGL backend could be imported. Attempting to create a "
                    "rendering context will result in a RuntimeError."
                )

        return opengl_context
