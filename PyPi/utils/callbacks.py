from PyPi.utils.dataset import max_QA


class CollectMaxQ(object):
    """
    This callback can be used to collect the values of the maximum action
    value in a given state at each call.
    """
    def __init__(self, approximator, state, action_values):
        """
        Constructor.

        Arguments
            approximator (object): the approximator to use;
            state (np.array): the state to consider;
            action_values (np.array): all the possible values of the action.
        """
        self._approximator = approximator
        self._state = state
        self._action_values = action_values

        self._max_Qs = list()

    def __call__(self):
        max_Q, _ = max_QA(self._state, False, self._approximator,
                          self._action_values)

        self._max_Qs.append(max_Q[0])

    def get_values(self):
        return self._max_Qs
