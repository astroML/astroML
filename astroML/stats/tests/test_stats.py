from __future__ import print_function, division

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal, assert_allclose)
from astroML.stats import (mean_sigma, median_sigmaG, sigmaG,
                           fit_bivariate_normal)
from astroML.stats.random import bivariate_normal


#---------------------------------------------------------------------------
# Check that mean_sigma() returns the same values as np.mean() and np.std()
def check_mean_sigma(a, axis=None, ddof=0):
    mu1, sigma1 = mean_sigma(a, axis=axis,
                             ddof=ddof)

    mu2 = np.mean(a, axis=axis)
    sigma2 = np.std(a, axis=axis, ddof=ddof)

    assert_array_almost_equal(mu1, mu2)
    assert_array_almost_equal(sigma1, sigma2)


def test_mean_sigma():
    np.random.seed(0)

    for shape in [(4, ), (4, 5), (4, 5, 6)]:
        a = np.random.random(shape)
        for axis in (None, 0):
            for ddof in (0, 1):
                yield (check_mean_sigma, a, axis, ddof)


#---------------------------------------------------------------------------
# Check that the keepdims argument works as expected
#  we'll later compare median_sigmaG to these results, so that
#  is effectively tested as well.
def check_mean_sigma_keepdims(a, axis):
    mu1, sigma1 = mean_sigma(a, axis, keepdims=False)
    mu2, sigma2 = mean_sigma(a, axis, keepdims=True)

    assert_array_equal(mu1.ravel(), mu2.ravel())
    assert_array_equal(sigma1.ravel(), sigma2.ravel())

    assert_array_equal(np.broadcast(a, mu2).shape, a.shape)
    assert_array_equal(np.broadcast(a, sigma2).shape, a.shape)


def test_mean_sigma_keepdims():
    np.random.seed(0)
    a = np.random.random((4, 5, 6))
    for axis in [None, 0, 1, 2]:
        yield (check_mean_sigma_keepdims, a, axis)


#---------------------------------------------------------------------------
# Check that median_sigmaG matches the values computed using np.percentile
# and np.median
def check_median_sigmaG(a, axis):
    from scipy.special import erfinv
    factor = 1. / (2 * np.sqrt(2) * erfinv(0.5))

    med1, sigmaG1 = median_sigmaG(a, axis=axis)
    med2 = np.median(a, axis=axis)
    q25, q75 = np.percentile(a, [25, 75], axis=axis)
    sigmaG2 = factor * (q75 - q25)

    assert_array_almost_equal(med1, med2)
    assert_array_almost_equal(sigmaG1, sigmaG2)


def test_median_sigmaG():
    np.random.seed(0)
    a = np.random.random((20, 40, 60))
    for axis in [None, 0, 1, 2]:
        yield (check_median_sigmaG, a, axis)


def check_sigmaG(a, axis):
    from scipy.special import erfinv
    factor = 1. / (2 * np.sqrt(2) * erfinv(0.5))

    sigmaG1 = sigmaG(a, axis=axis)
    q25, q75 = np.percentile(a, [25, 75], axis=axis)
    sigmaG2 = factor * (q75 - q25)

    assert_array_almost_equal(sigmaG1, sigmaG2)


def test_sigmaG():
    np.random.seed(0)
    a = np.random.random((20, 40, 60))
    for axis in [None, 0, 1, 2]:
        yield (check_sigmaG, a, axis)


#---------------------------------------------------------------------------
# Check that median_sigmaG() is a good approximation of mean_sigma()
# for normally-distributed data.
def check_median_sigmaG_approx(a, axis, keepdims, atol=0.15):
    med, sigmaG = median_sigmaG(a, axis=axis, keepdims=keepdims)
    mu, sigma = mean_sigma(a, axis=axis, ddof=1, keepdims=keepdims)

    assert_allclose(med, mu, atol=atol)
    assert_allclose(sigmaG, sigma, atol=atol)


def test_median_sigmaG_approx():
    np.random.seed(0)
    a = np.random.normal(0, 1, size=(10, 10000))
    for axis in (None, 1):
        for keepdims in (True, False):
            yield (check_median_sigmaG_approx, a, axis, keepdims, 0.02)


#---------------------------------------------------------------------------
# Check the bivariate normal fit
def check_fit_bivariate_normal(sigma1, sigma2, mu, alpha, N=1000):
    # poisson stats
    rtol = 2 * np.sqrt(N) / N

    x, y = bivariate_normal(mu, sigma1, sigma2, alpha, N).T
    mu_fit, sigma1_fit, sigma2_fit, alpha_fit = fit_bivariate_normal(x, y)

    if alpha_fit > np.pi / 2:
        alpha_fit -= np.pi
    elif alpha_fit < -np.pi / 2:
        alpha_fit += np.pi

    # Circular degeneracy in alpha: test sin(2*alpha) instead
    assert_allclose(np.sin(2 * alpha_fit), np.sin(2 * alpha), atol=2 * rtol)
    assert_allclose(mu, mu_fit, rtol=rtol)
    assert_allclose(sigma1_fit, sigma1, rtol=rtol)
    assert_allclose(sigma2_fit, sigma2, rtol=rtol)

def test_fit_bivariate_normal(sigma1=2.0, sigma2=1.0, N=1000):
    np.random.seed(0)
    mu = [10, 10]
    for alpha in np.linspace(-np.pi / 2, np.pi / 2, 7):
        yield check_fit_bivariate_normal, sigma1, sigma2, mu, alpha, N
