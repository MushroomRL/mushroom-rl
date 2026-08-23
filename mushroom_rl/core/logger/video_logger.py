import warnings

from mushroom_rl.utils.record import VideoRecorder


class VideoLogger(object):
    """
    This class implements the video recording functionality for the Logger.
    The recorder is created lazily on the first call to ``record_frame``.

    """
    def __init__(self, recorder_class=None, fps=None, video_path=None, append=False, **recorder_kwargs):
        """
        Constructor.

        Args:
            recorder_class (class, None): the class used to record the video. By default, the ``VideoRecorder`` class
                is used, and any class implementing its interface can be used instead;
            fps (int, None): frames per second for the video. If None, the default of the recorder class is used;
            video_path (Path, None): path where videos are stored. If None, videos go to the default location
                of the recorder class;
            append (bool, False): if True, the videos already present in ``video_path`` are loaded into the
                recorded videos list at construction;
            **recorder_kwargs: additional keyword arguments forwarded to the recorder class constructor.

        """
        self._recorder_class = recorder_class
        self._recorder_fps = fps
        self._video_path = video_path
        self._recorder_kwargs = recorder_kwargs
        self._recorder = None
        self._recorded_videos = list()

        if append:
            self._load_videos()

    def record_frame(self, frame, mask=None):
        """
        Record a single frame, or one frame per active environment.

        Args:
            frame: the frame to record (np.ndarray, H x W x RGB), the frames of the active environments
                (np.ndarray, N x H x W x RGB), or None when the environment drew nothing;
            mask (None): the mask of the active environments, telling which environment every frame belongs to.

        """
        if frame is None:
            warnings.warn('The environment drew nothing, the frame is not recorded.')

            return

        if frame.ndim == 4 and len(frame) == 1:
            frame = frame[0]

        if self._recorder is None:
            self._build_recorder(frame.ndim == 4)

        self._recorder(frame, mask)

    def stop_recording(self):
        """
        Stop the current recording. The next call to ``record_frame`` will start a new recording.

        Returns:
            The path of the recorded video file, or None if nothing was recorded.

        """
        if self._recorder is None:
            return None

        path = self._recorder.stop()

        if path is not None and path not in self._recorded_videos:
            self._recorded_videos.append(path)

        return path

    def set_video_fps(self, fps):
        """
        Set the frames per second for video recording, only if not already configured.

        Args:
            fps (int): frames per second.

        """
        if self._recorder_fps is None:
            self._recorder_fps = fps

    def _build_recorder(self, vectorized):
        kwargs = dict(fps=self._recorder_fps, path=self._video_path, vectorized=vectorized)
        kwargs.update(self._recorder_kwargs)

        recorder_class = self._recorder_class if self._recorder_class is not None else VideoRecorder

        self._recorder = recorder_class(**kwargs)

    def _load_videos(self):
        if self._video_path is not None and self._video_path.exists():
            self._recorded_videos = sorted(self._video_path.rglob('*.mp4'))

    @property
    def video_recorder(self):
        """
        Access to the underlying recorder instance. Returns None if no frame has been recorded yet.

        """
        return self._recorder

    @property
    def recorded_videos(self):
        """
        Returns:
            The list of paths of the videos recorded (and loaded) so far.

        """
        return self._recorded_videos
