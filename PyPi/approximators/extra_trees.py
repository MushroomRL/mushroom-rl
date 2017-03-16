from sklearn.ensemble import ExtraTreesRegressor


class ExtraTrees(object):
    def __init__(self, **approximator_params):
        self.approximator_params = approximator_params

    def fit(self, x, y, **fit_params):
        if not hasattr(self, 'model'):
            self.model = self._init()

        self.model.fit(x, y, **fit_params)

    def predict(self, x):
        predictions = self.model.predict(x)

        return predictions.ravel()

    def _init(self):
        return ExtraTreesRegressor(**self.approximator_params)
