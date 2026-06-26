import cv2
from pathlib import Path


class VideoRecorder(object):
    """
    Simple video record that creates a video from a stream of images.

    """

    def __init__(self, path="./mushroom_rl_recordings", video_name=None, fps=60, codec='vp09'):
        """
        Constructor.

        Args:
            path: Path at which videos will be stored.
            video_name: Name of the video without extension. Default is "recording".
            fps: Frame rate of the video.
            codec: FourCC codec string for the video writer. Default is "vp09" (VP9).
        """

        self._path = Path(path)

        self._video_name = video_name if video_name else "recording"
        self._counter = 0

        self._fps = fps
        self._codec = codec

        self._video_writer = None
        self._current_path = None

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
            name += f'-{self._counter}'
        name += '.mp4'

        self._path.mkdir(parents=True, exist_ok=True)

        self._current_path = self._path / name

        fourcc = cv2.VideoWriter_fourcc(*self._codec)
        self._video_writer = cv2.VideoWriter(str(self._current_path), fourcc, self._fps, (width, height))

    def stop(self):
        """
        Finalize the current video file.

        Returns:
            The path of the recorded video file, or None if nothing was recorded.

        """
        if self._video_writer is not None:
            cv2.destroyAllWindows()
            self._video_writer.release()
            self._video_writer = None
            self._counter += 1

        return self._current_path
