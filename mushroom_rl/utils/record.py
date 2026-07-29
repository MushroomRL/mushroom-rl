import cv2
from pathlib import Path


class VideoRecorder(object):
    """
    Simple video record that creates a video from a stream of images.

    """

    def __new__(cls, *args, vectorized=False, **kwargs):
        if cls is VideoRecorder and vectorized:
            return object.__new__(VectorizedVideoRecorder)
        else:
            return object.__new__(cls)

    def __init__(self, path=None, video_name=None, fps=None, codec='vp09', extension='.mp4', vectorized=False):
        """
        Constructor.

        Args:
            path: Path at which videos will be stored. Default is "./logs/videos".
            video_name: Name of the video without extension. Default is "recording".
            fps: Frame rate of the video. Default is 60.
            codec: FourCC codec string for the video writer. Default is "vp09" (VP9).
            extension: Extension of the video file. It must be a container supported by the codec.
            vectorized: If True, construction dispatches to the recorder of a vectorized environment.
        """

        self._path = Path(path if path else './logs/videos')

        self._video_name = video_name if video_name else "recording"
        self._counter = 0

        self._fps = fps if fps else 60
        self._codec = codec
        self._extension = extension

        self._video_writer = None
        self._current_path = None

    def __call__(self, frame, mask=None):
        """
        Args:
            frame (np.ndarray): Frame to be added to the video (H, W, RGB)
            mask (None): Mask of the active environments. Unused by this recorder, which records a single
                stream of frames.
        """
        assert frame is not None

        self._write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def stop(self):
        """
        Finalize the current video file.

        Returns:
            The path of the recorded video file, or None if nothing was recorded.

        """
        path = None

        if self._video_writer is not None:
            cv2.destroyAllWindows()
            self._video_writer.release()
            self._video_writer = None
            self._counter += 1
            path = self._current_path

        return path

    def _write(self, frame):
        if self._video_writer is None:
            height, width = frame.shape[:2]
            self._create_video_writer(height, width)

        self._video_writer.write(frame)

    def _create_video_writer(self, height, width):
        name = self._video_name
        if self._counter > 0:
            name += f'-{self._counter}'
        name += self._extension

        self._path.mkdir(parents=True, exist_ok=True)

        self._current_path = self._path / name

        fourcc = cv2.VideoWriter_fourcc(*self._codec)
        self._video_writer = cv2.VideoWriter(str(self._current_path), fourcc, self._fps, (width, height))


class VectorizedVideoRecorder(VideoRecorder):
    """
    Video recorder for a vectorized environment. Each environment is recorded in a separate temporary file, using a
    lossless codec, and the temporary files are concatenated into a single video when the recording is stopped: the
    resulting video shows one environment after the other. Since the temporary files are lossless, the final video
    is encoded only once, and has the same quality of a video recorded from a single environment.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._temp_path = self._path / f'.{self._video_name}_temp'
        self._recorders = dict()

    def __call__(self, frame, mask=None):
        """
        Args:
            frame (np.ndarray): Frames of the active environments (N, H, W, RGB), or a single frame (H, W, RGB)
                when the environment draws one image for all of them.
            mask (None): Mask of the active environments, telling which environment every frame belongs to.
                If None, the frames are assigned to the environments in order.
        """
        assert frame is not None

        if frame.ndim == 4:
            frames = iter(frame)

            for env, active in enumerate(mask if mask is not None else [True] * len(frame)):
                if active:
                    self._record_env_frame(next(frames), env)
        else:
            self._record_env_frame(frame, 0)

    def stop(self):
        """
        Finalize the current video file, concatenating the environments recorded so far, in order.

        Returns:
            The path of the recorded video file, or None if nothing was recorded.

        """
        for env in sorted(self._recorders):
            temp_path = self._recorders[env].stop()

            if temp_path is not None:
                self._append(temp_path)
                temp_path.unlink()

        self._recorders = dict()

        path = super().stop()

        if self._temp_path.exists() and not any(self._temp_path.iterdir()):
            self._temp_path.rmdir()

        return path

    def _record_env_frame(self, frame, env):
        if env not in self._recorders:
            self._recorders[env] = VideoRecorder(path=self._temp_path, video_name=f'env{env}', fps=self._fps,
                                                 codec='FFV1', extension='.mkv')

        self._recorders[env](frame)

    def _append(self, temp_path):
        capture = cv2.VideoCapture(str(temp_path))

        while True:
            read, frame = capture.read()

            if not read:
                break

            self._write(frame)

        capture.release()
