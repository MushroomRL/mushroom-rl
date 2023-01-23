import glfw
import mujoco
import time

import numpy as np


class MujocoGlfwViewer:
    """
    Class that creates a Glfw viewer for mujoco environments.
    Controls:
        Space: Pause / Unpause simulation
        c: Turn contact force and constraint visualisation on / off
        t: Make models transparent
    """
    def __init__(self, model, dt, width=1920, height=1080, start_paused=False):
        self.button_left = False
        self.button_right = False
        self.button_middle = False
        self.last_x = 0
        self.last_y = 0
        self.dt = dt

        self.frames = 0
        self.start_time = time.time()
        glfw.init()

        self._loop_count = 0
        self._time_per_render = 1 / 60.
        self._paused = start_paused

        self._window = glfw.create_window(width=width, height=height, title="MuJoCo", monitor=None, share=None)
        glfw.make_context_current(self._window)

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

        self._viewport = mujoco.MjrRect(0, 0, width, height)
        self._context = mujoco.MjrContext(model, mujoco.mjtFontScale(100))

        self.rgb_buffer = np.empty((width, height, 3), dtype=np.uint8)

    def mouse_button(self, window, button, act, mods):
        self.button_left = glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.button_right = glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        self.button_middle = glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS

        self.last_x, self.last_y = glfw.get_cursor_pos(self._window)

    def mouse_move(self, window, x_pos, y_pos):
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

    def scroll(self, window, x_offset, y_offset):
        mujoco.mjv_moveCamera(self._model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, 0.05 * y_offset, self._scene, self._camera)

    def render(self, data):

        def render_inner_loop(self):
            render_start = time.time()

            mujoco.mjv_updateScene(self._model, data, self._scene_option, None, self._camera,
                                   mujoco.mjtCatBit.mjCAT_ALL,
                                   self._scene)

            self._viewport.width, self._viewport.height = glfw.get_window_size(self._window)
            mujoco.mjr_render(self._viewport, self._scene, self._context)

            glfw.swap_buffers(self._window)
            glfw.poll_events()

            self.frames += 1

            if glfw.window_should_close(self._window):
                self.stop()
                exit(0)

            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (time.time() - render_start)
            """
            if return_img:
                mujoco.mjr_readPixels(self.rgb_buffer, None, self._viewport, self._context)
                return self.rgb_buffer
            """

        if self._paused:
            while self._paused:
                render_inner_loop(self)

        self._loop_count += self.dt / self._time_per_render
        while self._loop_count > 0:
            render_inner_loop(self)
            self._loop_count -= 1

    def stop(self):
        glfw.destroy_window(self._window)


