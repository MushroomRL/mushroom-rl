import mujoco


def forward_kinematics(mj_model, mj_data, q, body_name):
    """
    Compute the forward kinematics of the robots.

    Args:
        mj_model (mujoco.MjModel): mujoco MjModel of the robot-only model
        mj_data (mujoco.MjData): mujoco MjData object generated from the model
        q (np.array): joint configuration for which the forward kinematics are computed
        body_name (str): name of the body for which the fk is computed

    Returns (np.array(3), np.array(3, 3)):
        Position and Orientation of the body with the name body_name
    """

    mj_data.qpos[:len(q)] = q
    mujoco.mj_fwdPosition(mj_model, mj_data)
    return mj_data.body(body_name).xpos.copy(), mj_data.body(body_name).xmat.reshape(3, 3).copy()
