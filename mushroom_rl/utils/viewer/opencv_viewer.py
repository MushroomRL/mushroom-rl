import cv2

class CV2Viewer:
    """
    Simple viewer to display rendered images using cv2.

    """

    def __init__(self, window_name, dt, width, height):
        self._window_name = window_name
        self._dt = dt
        self._created_viewer = False
        self._width = width
        self._height = height

    def display(self, img):
        """
        Displays an image.

        Args:
            img (np.array): Image to display

        """

        # display image the first time
        if not self._created_viewer:
            # Removes toolbar and status bar
            cv2.namedWindow(self._window_name, flags=cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(self._window_name, self._width, self._height)
            cv2.imshow(self._window_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            self._wait()
            self._created_viewer = True

        # if the window is not closed yet, display another image
        elif not self._window_was_closed():
            cv2.imshow(self._window_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            self._wait()

        # window was closed, interrupt simulation
        else:
            exit()

    def _wait(self):
        """
        Wait for the specified amount of time. Time is supposed to be in milliseconds.

        """
        wait_time = int(self._dt * 1000)
        cv2.waitKey(wait_time)

    def _window_was_closed(self):
        """
        Check if a window was closed.

        Returns:
            True if the window was closed.

        """
        return cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE) == 0

    def close(self):
        if self._created_viewer:
            cv2.destroyWindow(self._window_name)