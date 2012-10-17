"""
Test density estimation techniques
"""
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm
from astroML.density_estimation import KDE, KNeighborsDensity


def test_KDE_gaussian_1D():
    np.random.seed(0)
    X = np.random.normal(0, 1, (5000, 1))
    X2 = np.linspace(-5, 5, 10).reshape((10, 1))

    clf = KDE('gaussian', h=0.1)
    clf.fit(X)
    dens = clf.eval(X2)

    true_dens = norm.pdf(X2[:, 0], 0, 1) * X.shape[0]

    # set atol very high: this will be a noisy result
    # note that this test could fail with a different random seed
    assert_allclose(dens, true_dens, atol=100)


def test_KDE_tophat_1D():
    np.random.seed(0)
    X = np.random.normal(0, 1, (5000, 1))
    X2 = np.linspace(-5, 5, 10).reshape((10, 1))

    clf = KDE('tophat', h=0.2)
    clf.fit(X)
    dens = clf.eval(X2)

    true_dens = norm.pdf(X2[:, 0], 0, 1) * X.shape[0]

    # set atol very high: this will be a noisy result
    # note that this test could fail with a different random seed
    assert_allclose(dens, true_dens, atol=100)


def test_KDE_expon_1D():
    np.random.seed(0)
    X = np.random.normal(0, 1, (5000, 1))
    X2 = np.linspace(-5, 5, 10).reshape((10, 1))

    clf = KDE('exponential', h=0.1)
    clf.fit(X)
    dens = clf.eval(X2)

    true_dens = norm.pdf(X2[:, 0], 0, 1) * X.shape[0]

    # set atol very high: this will be a noisy result
    # note that this test could fail with a different random seed
    assert_allclose(dens, true_dens, atol=100)


def test_KDE_quad_1D():
    np.random.seed(0)
    X = np.random.normal(0, 1, (5000, 1))
    X2 = np.linspace(-5, 5, 10).reshape((10, 1))

    clf = KDE('quadratic', h=0.2)
    clf.fit(X)
    dens = clf.eval(X2)

    true_dens = norm.pdf(X2[:, 0], 0, 1) * X.shape[0]

    # set atol very high: this will be a noisy result
    # note that this test could fail with a different random seed
    assert_allclose(dens, true_dens, atol=100)


def test_KNN_dens_simple_1D():
    np.random.seed(0)
    X = np.random.normal(0, 1, (10000, 1))
    X2 = np.linspace(-1, 1, 10).reshape((10, 1))

    clf = KNeighborsDensity(method='simple', n_neighbors=500)
    clf.fit(X)
    dens = clf.eval(X2)

    true_dens = norm.pdf(X2[:, 0], 0, 1) * X.shape[0]

    # set rtol & atol very high: this will be a noisy result
    # note that this test could fail with a different random seed
    assert_allclose(dens, true_dens, rtol=0.1, atol=100)


def test_KNN_dens_bayesian_1D():
    np.random.seed(0)
    X = np.random.normal(0, 1, (10000, 1))
    X2 = np.linspace(-1, 1, 10).reshape((10, 1))

    clf = KNeighborsDensity(method='bayesian', n_neighbors=500)
    clf.fit(X)
    dens = clf.eval(X2)

    true_dens = norm.pdf(X2[:, 0], 0, 1) * X.shape[0]

    # set rtol & atol very high: this will be a noisy result
    # note that this test could fail with a different random seed
    assert_allclose(dens, true_dens, rtol=0.1, atol=100)


def test_KNN_dens_bayesian_2D():
    np.random.seed(0)
    X = np.random.normal(0, 2, (100000, 2))

    grid = np.linspace(-1, 1, 10)
    X2 = np.array(np.meshgrid(grid, grid)).reshape((2, -1)).T

    clf = KNeighborsDensity(method='bayesian', n_neighbors=100)
    clf.fit(X)
    dens = clf.eval(X2)

    true_dens = norm.pdf(X2[:, 0], 0, 1) * X.shape[0]

    # set rtol & atol very high: this will be a noisy result
    # note that this test could fail with a different random seed
    assert_allclose(dens, true_dens, rtol=0.1, atol=100)
    
