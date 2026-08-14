import cv2
import numpy as np
from pathlib import Path
from pytest import raises

from mushroom_rl.utils.record import VideoRecorder, VectorizedVideoRecorder


def read_frame_values(path):
    capture = cv2.VideoCapture(str(path))
    values = list()

    while True:
        read, frame = capture.read()

        if not read:
            break

        values.append(frame.mean())

    capture.release()

    return values


def test_video_recorder_stop_without_frames(tmpdir):
    recorder = VideoRecorder(path=tmpdir, video_name='no_frames', fps=30)

    assert recorder.stop() is None
    assert not list(Path(tmpdir).rglob('*.mp4'))


def test_video_recorder_unknown_codec(tmpdir):
    recorder = VideoRecorder(path=tmpdir, video_name='unknown_codec', fps=30, codec='not_a_real_codec')

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with raises(ValueError):
        recorder(frame)


def test_video_recorder_codec_fails_to_open(tmpdir):
    recorder = VideoRecorder(path=tmpdir, video_name='unopenable_codec', fps=30, codec='mjpeg')

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with raises(ValueError):
        recorder(frame)


def test_video_recorder_writes_readable_video(tmpdir):
    recorder = VideoRecorder(path=tmpdir, video_name='readable', fps=30)

    for value in (10, 20, 30):
        recorder(np.full((100, 100, 3), value, dtype=np.uint8))

    path = recorder.stop()

    assert path == Path(tmpdir) / 'readable.mp4'
    assert path.stat().st_size > 0

    values = read_frame_values(path)

    assert len(values) == 3
    assert all(value > 0 for value in values)


def test_vectorized_video_recorder_temp_files_use_ffv1(tmpdir):
    recorder = VectorizedVideoRecorder(path=tmpdir, video_name='ffv1_check', fps=30)

    frame = np.zeros((1, 100, 100, 3), dtype=np.uint8)
    recorder(frame, np.array([True]))

    assert recorder._recorders[0]._codec == 'ffv1'

    recorder.stop()
