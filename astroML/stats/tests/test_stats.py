import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose)
from astroML.stats import (mean_sigma, median_sigmaG, sigmaG,
                           fit_bivariate_normal)
from astroML.stats.random import bivariate_normal, trunc_exp, linear


#---------------------------------------------------------------------------
# Check that mean_sigma() returns the same values as np.mean() and np.std()
@pytest.mark.parametrize("a_shape", [(4, ), (4, 5), (4, 5, 6)])
@pytest.mark.parametrize("axis", [None, 0])
@pytest.mark.parametrize("ddof", [0, 1])
def test_mean_sigma(a_shape, axis, ddof):
    np.random.seed(0)

    a = np.random.random(a_shape)
    mu1, sigma1 = mean_sigma(a, axis=axis,
                             ddof=ddof)

    mu2 = np.mean(a, axis=axis)
    sigma2 = np.std(a, axis=axis, ddof=ddof)

    assert_array_almost_equal(mu1, mu2)
    assert_array_almost_equal(sigma1, sigma2)


#---------------------------------------------------------------------------
# Check that the keepdims argument works as expected
#  we'll later compare median_sigmaG to these results, so that
#  is effectively tested as well.
@pytest.mark.parametrize("axis", [None, 0, 1, 2])
def test_mean_sigma_keepdims(axis):
    np.random.seed(0)
    a = np.random.random((4, 5, 6))
    mu1, sigma1 = mean_sigma(a, axis, keepdims=False)
    mu2, sigma2 = mean_sigma(a, axis, keepdims=True)

    assert_array_equal(mu1.ravel(), mu2.ravel())
    assert_array_equal(sigma1.ravel(), sigma2.ravel())

    assert_array_equal(np.broadcast(a, mu2).shape, a.shape)
    assert_array_equal(np.broadcast(a, sigma2).shape, a.shape)


#---------------------------------------------------------------------------
# Check that median_sigmaG matches the values computed using np.percentile
# and np.median
@pytest.mark.parametrize("axis", [None, 0, 1, 2])
def test_median_sigmaG(axis):
    np.random.seed(0)
    a = np.random.random((20, 40, 60))

    from scipy.special import erfinv
    factor = 1. / (2 * np.sqrt(2) * erfinv(0.5))

    med1, sigmaG1 = median_sigmaG(a, axis=axis)
    med2 = np.median(a, axis=axis)
    q25, q75 = np.percentile(a, [25, 75], axis=axis)
    sigmaG2 = factor * (q75 - q25)

    assert_array_almost_equal(med1, med2)
    assert_array_almost_equal(sigmaG1, sigmaG2)


@pytest.mark.parametrize("axis", [None, 0, 1, 2])
def test_sigmaG(axis):
    np.random.seed(0)
    a = np.random.random((20, 40, 60))

    from scipy.special import erfinv
    factor = 1. / (2 * np.sqrt(2) * erfinv(0.5))

    sigmaG1 = sigmaG(a, axis=axis)
    q25, q75 = np.percentile(a, [25, 75], axis=axis)
    sigmaG2 = factor * (q75 - q25)

    assert_array_almost_equal(sigmaG1, sigmaG2)


#---------------------------------------------------------------------------
# Check that median_sigmaG() is a good approximation of mean_sigma()
# for normally-distributed data.
@pytest.mark.parametrize('axis', [None, 1])
@pytest.mark.parametrize('keepdims', [True, False])
def test_median_sigmaG_approx(axis, keepdims, atol=0.02):
    np.random.seed(0)
    a = np.random.normal(0, 1, size=(10, 10000))

    med, sigmaG = median_sigmaG(a, axis=axis, keepdims=keepdims)
    mu, sigma = mean_sigma(a, axis=axis, ddof=1, keepdims=keepdims)

    assert_allclose(med, mu, atol=atol)
    assert_allclose(sigmaG, sigma, atol=atol)


#---------------------------------------------------------------------------
# Check the bivariate normal fit

@pytest.mark.parametrize("alpha", np.linspace(-np.pi / 2, np.pi / 2, 7))
def test_fit_bivariate_normal(alpha):
    mu = [10, 10]

    sigma1 = 2.0
    sigma2 = 1.0
    N = 1000

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


#------------------------------------------------------
# Check truncated exponential and linear functions
def test_trunc_exp():
    x = np.linspace(0, 10, 100)
    k = 0.25
    xlim = [3, 5]
    # replaced with from astroML.stats.random import trunc_exp
    # trunc_exp = trunc_exp_gen(name="trunc_exp", shapes='a, b, k')
    myfunc = trunc_exp(xlim[0], xlim[1], k)
    y = myfunc.pdf(x)
    zeros = np.zeros(len(y))

    # Test that the function is zero outside of defined limits
    assert_array_equal(y[x < xlim[0]], zeros[x < xlim[0]])
    assert_array_equal(y[x > xlim[1]], zeros[x > xlim[1]])
    inlims = (x < xlim[1]) & (x > xlim[0])
    C = k / (np.exp(k * xlim[1]) - np.exp(k * xlim[0]))

    # Test that within defined limits, function is exponential
    assert_array_equal(y[inlims], C*np.exp(k * x[inlims]))

    # Test that the PDF integrates to just about 1
    dx = x[1] - x[0]
    integral = np.sum(y * dx)
    assert np.round(integral, 1) == 1


# Check the linear generator
def test_linear_gen():
    x = np.linspace(-10, 10, 200)
    c = -0.5
    xlim = [-2.4, 6.]
    # replaced with from astroML.stats.random import linear
    # linear = linear_gen(name="linear", shapes="a, b, c")
    y = linear.pdf(x, xlim[0], xlim[1], c)
    zeros = np.zeros(len(y))

    # Test that the function is zero outside of defined limits
    assert_array_equal(y[x < xlim[0]], zeros[x < xlim[0]])
    assert_array_equal(y[x > xlim[1]], zeros[x > xlim[1]])
    inlims = (x < xlim[1]) & (x > xlim[0])
    d = 1. / (xlim[1] - xlim[0]) - 0.5 * c * (xlim[1] + xlim[0])
    inlims = (x < xlim[1]) & (x > xlim[0])

    # Test that within defined limits, function is linear
    assert_array_equal(y[inlims], c*x[inlims] + d)

    # Test that the PDF integrates to about 1
    dx = x[1] - x[0]
    integral = np.sum(y * dx)
    assert np.round(integral, 1) == 1
