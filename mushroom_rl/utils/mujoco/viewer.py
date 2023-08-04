import glfw
import mujoco
import time
from itertools import cycle

import numpy as np


class MujocoGlfwViewer:
    """
    Class that creates a Glfw viewer for mujoco environments.

    """

    def __init__(self, model, dt, width=1920, height=1080, start_paused=False,
                 custom_render_callback=None, record=False, camera_params=None,
                 default_camera_mode="static", hide_menu_on_startup=False):
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

        """

        self.button_left = False
        self.button_right = False
        self.button_middle = False
        self.last_x = 0
        self.last_y = 0
        self.dt = dt

        self.frames = 0
        self.start_time = time.time()
        glfw.init()
        glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, 0)

        if record:
            # dont allow to change the window size to have equal frame size during recording
            glfw.window_hint(glfw.RESIZABLE, False)

        self._loop_count = 0
        self._time_per_render = 1 / 60.
        self._run_speed_factor = 1.0
        self._paused = start_paused

        self._window = glfw.create_window(width=width, height=height, title="MuJoCo", monitor=None, share=None)
        glfw.make_context_current(self._window)

        self._width = width
        self._height = height

        # Disable v_sync, so swap_buffers does not block
        # glfw.swap_interval(0)

        glfw.set_mouse_button_callback(self._window, self.mouse_button)
        glfw.set_cursor_pos_callback(self._window, self.mouse_move)
        glfw.set_key_callback(self._window, self.keyboard)
        glfw.set_scroll_callback(self._window, self.scroll)

        self._model = model

        self._scene = mujoco.MjvScene(model, 1000)
        self._scene_option = mujoco.MjvOption()

        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self._camera)
        if camera_params is None:
            self._camera_params = self.get_default_camera_params()
        else:
            self._camera_params = self._assert_camera_params(camera_params)
        self._all_camera_modes = ("static", "follow", "top_static")
        self._camera_mode_iter = cycle(self._all_camera_modes)
        self._camera_mode = next(self._camera_mode_iter)
        self._camera_mode_target = self._camera_mode
        assert default_camera_mode in self._all_camera_modes
        while self._camera_mode_target != default_camera_mode:
            self._camera_mode_target = next(self._camera_mode_iter)
        self._set_camera()

        self._viewport = mujoco.MjrRect(0, 0, width, height)
        self._font_scale = 100
        self._context = mujoco.MjrContext(model, mujoco.mjtFontScale(self._font_scale))

        self.custom_render_callback = custom_render_callback

        self._overlay = {}
        self._hide_menu = hide_menu_on_startup

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

        if key == glfw.KEY_TAB:
            self._camera_mode_target = next(self._camera_mode_iter)

        if key == glfw.KEY_S:
            self._run_speed_factor /= 2.0

        if key == glfw.KEY_F:
            self._run_speed_factor *= 2.0

        if key == glfw.KEY_G:
            for i in range(len(self._scene_option.geomgroup)):
                self._scene_option.geomgroup[i] = not self._scene_option.geomgroup[i]

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

            self._create_overlay()

            render_start = time.time()

            mujoco.mjv_updateScene(self._model, data, self._scene_option, None, self._camera,
                                   mujoco.mjtCatBit.mjCAT_ALL,
                                   self._scene)

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

            glfw.swap_buffers(self._window)
            glfw.poll_events()

            self.frames += 1

            self._overlay.clear()

            if glfw.window_should_close(self._window):
                self.stop()
                exit(0)

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
            "[S]lower, [F]aster")
        
        add_overlay(
            topleft,
            "Press E to toggle reference frames.")
        
        add_overlay(
            topleft,
            "Press G to toggle geom groups.",
            make_new_line=False)

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

        return dict(static=dict(distance=15.0, elevation=-45.0, azimuth=90.0),
                    follow=dict(distance=3.5, elevation=0.0, azimuth=90.0),
                    top_static=dict(distance=5.0, elevation=-90.0, azimuth=90.0))
