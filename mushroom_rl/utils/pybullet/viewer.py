import numpy as np
import pybullet
from mushroom_rl.utils.viewer import ImageViewer


class PyBulletViewer(ImageViewer):
    def __init__(self, client, dt, size=(500, 500), distance=4, origin=(0, 0, 1), angles=(0, -45, 60),
                 fov=60, aspect=1, near_val=0.01, far_val=100):
        self._client = client
        self._size = size
        self._distance = distance
        self._origin = origin
        self._angles = angles
        self._fov = fov
        self._aspect = aspect
        self._near_val = near_val
        self._far_val = far_val
        super().__init__(size, dt)

    def display(self):
        img = self._get_image()
        super().display(img)

    def _get_image(self):
        view_matrix = self._client.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=self._origin,
                                                                     distance=self._distance,
                                                                     roll=self._angles[0],
                                                                     pitch=self._angles[1],
                                                                     yaw=self._angles[2],
                                                                     upAxisIndex=2)
        proj_matrix = self._client.computeProjectionMatrixFOV(fov=self._fov, aspect=self._aspect,
                                                              nearVal=self._near_val, farVal=self._far_val)
        (_, _, px, _, _) = self._client.getCameraImage(width=self._size[0],
                                                       height=self._size[1],
                                                       viewMatrix=view_matrix,
                                                       projectionMatrix=proj_matrix,
                                                       renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.reshape(np.array(px), (self._size[0], self._size[1], -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array
