import os
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
            fps: Frame rate of the video.
        """

        if not os.path.isdir(path):
            os.mkdir(path)

        if tag is None:
            date_time = datetime.datetime.now()
            tag = date_time.strftime("%d-%m-%Y_%H-%M-%S")

        path = path + "/" + tag

        if not os.path.isdir(path):
            os.mkdir(path)

        if video_name:
            suffix = Path(video_name).suffix
            if suffix == "":
                video_name += ".mp4"
            elif suffix != ".mp4":
                raise ValueError("Provided video name has unsupported suffix \"%s\"! "
                                 "Please use \".mp4\" or don't provide suffix." % suffix)

            path += "/" + video_name
        else:
            path += "/" + "recording.mp4"

        self._path = path

        self._fps = fps

        self._video_writer = None

    def __call__(self, frame):
        """
        Args:
            frame (np.ndarray): Frame to be added to the video (H, W, RGB)
        """
        if self._video_writer is None:
            height, width = frame.shape[:2]
            self._video_writer = cv2.VideoWriter(self._path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                                 self._fps, (width, height))

        self._video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def stop(self):
        cv2.destroyAllWindows()
        self._video_writer.release()