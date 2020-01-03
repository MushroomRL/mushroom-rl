from mushroom_rl.utils.table import Table


def EligibilityTrace(shape, name='replacing'):
    """
    Factory method to create an eligibility trace of the provided type.

    Args:
        shape (list): shape of the eligibility trace table;
        name (str, 'replacing'): type of the eligibility trace.

    Returns:
        The eligibility trace table of the provided shape and type.

    """
    if name == 'replacing':
        return ReplacingTrace(shape)
    elif name == 'accumulating':
        return AccumulatingTrace(shape)
    else:
        raise ValueError('Unknown type of trace.')


class ReplacingTrace(Table):
    """
    Replacing trace.

    """
    def reset(self):
        self.table[:] = 0.

    def update(self, state, action):
        self.table[state, action] = 1.


class AccumulatingTrace(Table):
    """
    Accumulating trace.

    """
    def reset(self):
        self.table[:] = 0.

    def update(self, state, action):
        self.table[state, action] += 1.
