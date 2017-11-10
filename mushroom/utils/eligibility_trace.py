from mushroom.utils.table import Table


def EligibilityTrace(shape, name='replacing'):
    if name == 'replacing':
        return ReplacingTrace(shape)
    elif name == 'accumulating':
        return AccumulatingTrace(shape)
    else:
        raise ValueError('Unknown type of trace.')


class ReplacingTrace(Table):
    def reset(self):
        self.table[:] = 0.

    def update(self, state, action):
        self.table[state, action] = 1.


class AccumulatingTrace(Table):
    def reset(self):
        self.table[:] = 0.

    def update(self, state, action):
        self.table[state, action] += 1.
