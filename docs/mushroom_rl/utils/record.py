import cv2
import datetime
from pathlib import Path


class VideoRecorder(object):
    """
    Simple video record that creates a video from a stream of images.

    """

    def __init__(self, path="./mushroom_rl_recordings", tag=None, video_name=None, fps=60):
        """
        Constructor.

        Args:
            path: Path at which videos will be stored.
            tag: Name of the directory at path in which the video will be stored. If None, a timestamp will be created.
            video_name: Name of the video without extension. Default is "recording".
            fps: Frame rate of the video.
        """

        if tag is None:
            date_time = datetime.datetime.now()
            tag = date_time.strftime("%d-%m-%Y_%H-%M-%S")

        self._path = Path(path)
        self._path = self._path / tag

        self._video_name = video_name if video_name else "recording"
        self._counter = 0

        self._fps = fps

        self._video_writer = None

    def __call__(self, frame):
        """
        Args:
            frame (np.ndarray): Frame to be added to the video (H, W, RGB)
        """
        assert frame is not None

        if self._video_writer is None:
            height, width = frame.shape[:2]
            self._create_video_writer(height, width)

        self._video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def _create_video_writer(self, height, width):

        name = self._video_name
        if self._counter > 0:
            name += f"-{self._counter}.mp4"
        else:
            name += ".mp4"

        self._path.mkdir(parents=True, exist_ok=True)

        path = self._path / name

        self._video_writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                             self._fps, (width, height))

    def stop(self):
        cv2.destroyAllWindows()
        self._video_writer.release()
        self._video_writer = None
        self._counter += 1
