from mushroom_rl.approximators.table import Table
from mushroom_rl.approximators.ensemble import Ensemble


class EnsembleTable(Ensemble):
    """
    This class implements functions to manage table ensembles.

    """
    def __init__(self, n_models, shape, **params):
        """
        Constructor.

        Args:
            n_models (int): number of models in the ensemble;
            shape (np.ndarray): shape of each table in the ensemble.
            **params: parameters dictionary to create each regressor.

        """
        params['shape'] = shape
        super().__init__(Table, n_models, **params)

    @property
    def n_actions(self):
        return self._model[0].shape[-1]