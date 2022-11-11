import numpy as np

from mushroom_rl.algorithms.value.batch_td import BatchTD
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.features import get_action_features
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.utils.parameters import to_parameter


class LSPI(BatchTD):
    """
    Least-Squares Policy Iteration algorithm.
    "Least-Squares Policy Iteration". Lagoudakis M. G. and Parr R.. 2003.

    """
    def __init__(self, mdp_info, policy, approximator_params=None,
                 epsilon=1e-2, fit_params=None, features=None):
        """
        Constructor.

        Args:
            epsilon ([float, Parameter], 1e-2): termination coefficient.

        """
        self._epsilon = to_parameter(epsilon)

        self._add_save_attr(_epsilon='mushroom')

        super().__init__(mdp_info, policy, LinearApproximator,
                         approximator_params, fit_params, features)

    def fit(self, dataset, **info):
        phi_state, action, reward, phi_next_state, absorbing, _ = parse_dataset(
            dataset, self.phi)
        phi_state_action = get_action_features(phi_state, action,
                                               self.mdp_info.action_space.n)

        norm = np.inf
        while norm > self._epsilon():
            q = self.approximator.predict(phi_next_state)
            if np.any(absorbing):
                q *= 1 - absorbing.reshape(-1, 1)

            next_action = np.argmax(q, axis=1).reshape(-1, 1)
            phi_next_state_next_action = get_action_features(
                phi_next_state,
                next_action,
                self.mdp_info.action_space.n
            )

            tmp = phi_state_action - self.mdp_info.gamma *\
                phi_next_state_next_action
            A = phi_state_action.T.dot(tmp)
            b = (phi_state_action.T.dot(reward)).reshape(-1, 1)

            old_w = self.approximator.get_weights()
            if np.linalg.matrix_rank(A) == A.shape[1]:
                w = np.linalg.solve(A, b).ravel()
            else:
                w = np.linalg.pinv(A).dot(b).ravel()
            self.approximator.set_weights(w)

            norm = np.linalg.norm(w - old_w)
