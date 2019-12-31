"""
Test density estimation techniques
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm
from astroML.density_estimation import KNeighborsDensity, GaussianMixture1D


classifiers = [KNeighborsDensity(method='simple', n_neighbors=250),
               KNeighborsDensity(method='bayesian', n_neighbors=250)]


@pytest.mark.parametrize("clf", classifiers)
def test_1D_density(clf, atol=100):
    np.random.seed(0)
    dist = norm(0, 1)

    X = dist.rvs((5000, 1))
    X2 = np.linspace(-5, 5, 10).reshape((10, 1))
    true_dens = dist.pdf(X2[:, 0]) * X.shape[0]

    clf.fit(X)
    dens = clf.eval(X2)

    assert_allclose(dens, true_dens, atol=atol)


def test_gaussian1d():
    x = np.linspace(-6, 10, 1000)
    means = np.array([-1.5, 0.0, 2.3])
    sigmas = np.array([1, 0.25, 3.8])
    weights = np.array([1, 1, 1])

    gauss = GaussianMixture1D(means=means, sigmas=sigmas, weights=weights)
    y = gauss.pdf(x)

    # Check whether sampling works
    gauss.sample(10)

    dx = x[1] - x[0]
    integral = np.sum(y*dx)

    assert_allclose(integral, 1., atol=0.02)
