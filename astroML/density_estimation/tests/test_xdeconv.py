import numpy as np
from numpy.testing import assert_allclose
from astroML.density_estimation import XDGMM


def test_XDGMM_1D_gaussian(N=100, sigma=0.1):
    np.random.seed(0)
    mu = 0
    V = 1

    X = np.random.normal(mu, V, size=(N, 1))
    X += np.random.normal(0, sigma, size=(N, 1))
    Xerr = sigma ** 2 * np.ones((N, 1, 1))

    xdgmm = XDGMM(1).fit(X, Xerr)

    # because of sample variance, results will be similar
    # but not identical.  We'll use a fudge factor of 0.1
    assert_allclose(mu, xdgmm.mu[0], atol=0.1)
    assert_allclose(V, xdgmm.V[0], atol=0.1)


def check_single_gaussian(N=100, D=3, sigma=0.1):
    np.random.seed(0)
    mu = np.random.random(D)
    V = np.random.random((D, D))
    V = np.dot(V, V.T)

    X = np.random.multivariate_normal(mu, V, size=N)
    Xerr = np.zeros((N, D, D))
    Xerr[:, range(D), range(D)] = sigma ** 2

    X += np.random.normal(0, sigma, X.shape)

    xdgmm = XDGMM(1)
    xdgmm.fit(X, Xerr)

    # because of sample variance, results will be similar
    # but not identical.  We'll use a fudge factor of 0.1
    assert_allclose(mu, xdgmm.mu[0], atol=0.1)
    assert_allclose(V, xdgmm.V[0], atol=0.1)


def test_single_gaussian(N=100, sigma=0.1):
    for D in (1, 2, 3):
        yield (check_single_gaussian, N, D, sigma)
