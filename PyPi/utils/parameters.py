class Parameter(object):
    def __init__(self, value, decay_type=None, decay_factor=1, min_value=None):
        self.value = value
        self.decay_type = decay_type
        self.decay_factor = decay_factor
        self.min_value = min_value

    def __call__(self):
        return self.value

    def update(self):
        if self.decay_type is None:
            return
        elif self.decay_type == 'linear':
            self.value -= self.decay_factor
        elif self.decay_type == 'exponential':
            self.value *= self.decay_factor
        else:
            raise ValueError('Selected decay_type not available')
