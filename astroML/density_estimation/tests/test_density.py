"""
Test density estimation techniques
"""
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm
from astroML.density_estimation import KDE, KNeighborsDensity


def check_1D_density(clf, X, X2, true_dens, atol):
    clf.fit(X)
    dens = clf.eval(X2)

    assert_allclose(dens, true_dens, atol=atol)


def test_1D_density():
    np.random.seed(0)
    dist = norm(0, 1)

    X = dist.rvs((5000, 1))
    X2 = np.linspace(-5, 5, 10).reshape((10, 1))
    true_dens = dist.pdf(X2[:, 0]) * X.shape[0]

    classifiers = [KDE('gaussian', h=0.1),
                   KDE('tophat', h=0.2),
                   KDE('exponential', h=0.1),
                   KDE('quadratic', h=0.2),
                   KNeighborsDensity(method='simple', n_neighbors=250),
                   KNeighborsDensity(method='bayesian', n_neighbors=250)]

    for clf in classifiers:
        yield (check_1D_density, clf, X, X2, true_dens, 100)
