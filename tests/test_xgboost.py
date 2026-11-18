import numpy as np

from vas.train import _fit_xgboost


def test_xgboost_search_with_groups():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    groups = np.repeat(np.arange(3), 100)
    model = _fit_xgboost(X, y, np.ones(len(y)), groups, seed=0, n_iter=2, n_folds=3)
    assert (model.predict(X) == y).mean() > 0.9
