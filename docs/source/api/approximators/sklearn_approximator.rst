Scikit-learn approximator
=========================

``SklearnApproximator`` wraps a scikit-learn estimator class as a numpy approximator. The estimator class is passed as
``model_class`` and every other unrecognized parameter is forwarded to its constructor:

.. code-block:: python

    from sklearn.ensemble import ExtraTreesRegressor

    approximator_params = dict(model_class=ExtraTreesRegressor, input_shape=(2,), n_actions=2,
                               n_estimators=50, min_samples_split=5, min_samples_leaf=2)
    agent = FQI(mdp.info, pi, SklearnApproximator, n_iterations=20, approximator_params=approximator_params)

.. automodule:: mushroom_rl.approximators.sklearn_approximator
