from mushroom_rl.core import Serializable


class ContextBuilder(Serializable):
    def __init__(self, context_shape=None):
        self._context_shape = context_shape

        super().__init__()

        self._add_save_attr(_context_shape='primitive')

    def __call__(self, initial_state, **episode_info):
        return None

    @property
    def context_shape(self):
        return self._context_shape
