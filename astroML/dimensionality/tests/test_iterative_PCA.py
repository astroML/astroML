import numpy as np
from numpy.testing import assert_array_almost_equal
from astroML.dimensionality import iterative_pca


def test_iterative_PCA(n_samples=50, n_features=40):
    np.random.seed(0)

    # construct some data that is well-approximated
    #  by two principal components
    x = np.linspace(0, np.pi, n_features)
    x0 = np.linspace(0, np.pi, n_samples)
    X = np.sin(x) * np.cos(0.5 * (x - x0[:, None]))

    # mask 10% of the pixels
    M = (np.random.random(X.shape) > 0.9)

    # reconstruct and check accuracy
    for norm in (None, 'L1', 'L2'):
        X_recons = iterative_pca(X, M, n_ev=2, n_iter=10, norm=norm)

        assert_array_almost_equal(X, X_recons, decimal=2)
