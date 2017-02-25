class Regressor(object):
    def __init__(self, approximator_class, **apprx_params):
        self.model = approximator_class(**apprx_params)

    def fit(self, x, y):
        # estrarre features, normalizzare e fit
        self.model.fit(x, y)

    def predict(self, x):
        # estrarre features, normalizzare e predict
        return self.model.predict(x)
