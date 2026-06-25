from mushroom_rl.utils.record import VideoRecorder


class VideoLogger(object):
    """
    This class implements the video recording functionality for the Logger.
    The recorder is created lazily on the first call to ``record_frame``.

    """
    def __init__(self, recorder_class=None, fps=None, video_path=None, **recorder_kwargs):
        """
        Constructor.

        Args:
            recorder_class (class, None): the class used to record the video. By default, the ``VideoRecorder`` class
                is used. The class must implement the ``__call__`` and ``stop`` methods;
            fps (int, None): frames per second for the video. If None, the default of the recorder class is used;
            video_path (Path, None): path where videos are stored. If None, videos go to the default location
                of the recorder class;
            **recorder_kwargs: additional keyword arguments forwarded to the recorder class constructor.

        """
        self._recorder_class = recorder_class
        self._recorder_fps = fps
        self._video_path = video_path
        self._recorder_kwargs = recorder_kwargs
        self._recorder = None

    def record_frame(self, frame):
        """
        Record a single frame. The recorder is created lazily on the first call.

        Args:
            frame: the frame to record (np.ndarray, H x W x RGB).

        """
        if self._recorder is None:
            self._build_recorder()

        self._recorder(frame)

    def stop_recording(self):
        """
        Stop the current recording. The next call to ``record_frame`` will start a new recording.

        """
        if self._recorder is not None:
            self._recorder.stop()

    def set_video_fps(self, fps):
        """
        Set the frames per second for video recording, only if not already configured.

        Args:
            fps (int): frames per second.

        """
        if self._recorder_fps is None:
            self._recorder_fps = fps

    def _build_recorder(self):
        kwargs = dict(self._recorder_kwargs)

        if self._recorder_fps is not None:
            kwargs['fps'] = self._recorder_fps

        recorder_class = self._recorder_class if self._recorder_class is not None else VideoRecorder

        if recorder_class is VideoRecorder and self._video_path is not None:
            kwargs['path'] = str(self._video_path)

        self._recorder = recorder_class(**kwargs)

    @property
    def recorder(self):
        """
        Access to the underlying recorder instance. Returns None if no frame has been recorded yet.

        """
        return self._recorder
