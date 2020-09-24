import numpy as np
import cv2

cv2.ocl.setUseOpenCL(False)


class LazyFrames(object):
    """
    From OpenAI Baseline.
    https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

    """
    def __init__(self, frames, history_length):
        self._frames = frames

        assert len(self._frames) == history_length

    def __array__(self, dtype=None):
        out = np.array(self._frames)
        if dtype is not None:
            out = out.astype(dtype)

        return out

    def copy(self):
        return self

    @property
    def shape(self):
        return (len(self._frames),) + self._frames[0].shape


def preprocess_frame(obs, img_size):
    """
    Convert a frame from rgb to grayscale and resize it.

    Args:
        obs (np.ndarray): array representing an rgb frame;
        img_size: target size for images.

    Returns:
        The transformed frame as 8 bit integer array.

    """
    image = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, img_size, interpolation=cv2.INTER_LINEAR)

    return np.array(image, dtype=np.uint8)
