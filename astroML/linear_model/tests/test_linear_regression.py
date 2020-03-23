import pytest

import numpy as np
from numpy.testing import assert_allclose

from sklearn.linear_model import LinearRegression as skLinearRegression
from astroML.linear_model import \
    LinearRegression, PolynomialRegression, BasisFunctionRegression

try:
    import pymc3 as pm
    HAS_PYMC3 = True
except ImportError:
    HAS_PYMC3 = False


def test_error_transform_diag(N=20, rseed=0):
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 2)
    yerr = 0.05 * (1 + rng.rand(N))
    y = (X[:, 0] ** 2 + X[:, 1]) + yerr * rng.randn(N)
    Sigma = np.eye(N) * yerr ** 2

    X1, y1 = LinearRegression._scale_by_error(X, y, yerr)
    X2, y2 = LinearRegression._scale_by_error(X, y, Sigma)

    assert_allclose(X1, X2)
    assert_allclose(y1, y2)


def test_error_transform_full(N=20, rseed=0):
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 2)

    # generate a pos-definite error matrix
    Sigma = 0.05 * rng.randn(N, N)
    u, s, v = np.linalg.svd(Sigma)
    Sigma = np.dot(u * s, u.T)

    # draw y from this error distribution
    y = (X[:, 0] ** 2 + X[:, 1])
    y = rng.multivariate_normal(y, Sigma)

    X2, y2 = LinearRegression._scale_by_error(X, y, Sigma)

    # check that the form entering the chi^2 is correct
    assert_allclose(np.dot(X2.T, X2),
                    np.dot(X.T, np.linalg.solve(Sigma, X)))
    assert_allclose(np.dot(y2, y2),
                    np.dot(y, np.linalg.solve(Sigma, y)))


def test_LinearRegression_simple():
    """
    Test a simple linear regression
    """
    x = np.arange(10.).reshape((10, 1))
    y = np.arange(10.) + 1
    dy = 1

    clf = LinearRegression().fit(x, y, dy)
    y_true = clf.predict(x)

    assert_allclose(y, y_true, atol=1E-10)


def test_LinearRegression_err():
    """
    Test that errors are correctly accounted for
    By comparing to scikit-learn LinearRegression
    """
    np.random.seed(0)
    X = np.random.random((10, 1))
    y = np.random.random(10) + 1
    dy = 0.1

    y = np.random.normal(y, dy)

    X_fit = np.linspace(0, 1, 10)[:, None]
    clf1 = LinearRegression().fit(X, y, dy)
    clf2 = skLinearRegression().fit(X / dy, y / dy)

    assert_allclose(clf1.coef_[1:], clf2.coef_)
    assert_allclose(clf1.coef_[0], clf2.intercept_ * dy)


def test_LinearRegression_fit_intercept():
    np.random.seed(0)
    X = np.random.random((10, 1))
    y = np.random.random(10)

    clf1 = LinearRegression(fit_intercept=False).fit(X, y)
    clf2 = skLinearRegression(fit_intercept=False).fit(X, y)

    assert_allclose(clf1.coef_, clf2.coef_)


def test_PolynomialRegression_simple():
    x = np.arange(10.).reshape((10, 1))
    y = np.arange(10.)
    dy = 1

    clf = PolynomialRegression(2).fit(x, y, dy)
    y_true = clf.predict(x)

    assert_allclose(y, y_true, atol=1E-10)


def test_BasisfunctionRegression_simple():
    x = np.arange(10.).reshape((10, 1))
    y = np.arange(10.) + 1
    dy = 1

    mu = np.arange(11.)[:, None]
    sigma = 1.0

    clf = BasisFunctionRegression(mu=mu, sigma=sigma).fit(x, y, dy)
    y_true = clf.predict(x)

    assert_allclose(y, y_true, atol=1E-10)


@pytest.mark.skipif('not HAS_PYMC3')
def test_LinearRegressionwithErrors():
    """
    Test for small errors agrees with fit with y errors only
    """

    from astroML.linear_model import LinearRegressionwithErrors

    np.random.seed(0)
    X = np.random.random(10) + 1
    dy = np.random.random(10) * 0.1
    y = X * 2 + 1 + (dy - 0.05)
    dx = np.random.random(10) * 0.01
    X = X + (dx - 0.005)

    clf1 = LinearRegression().fit(X[:, None], y, dy)
    clf2 = LinearRegressionwithErrors().fit(np.atleast_2d(X), y, dy, dx)

    assert_allclose(clf1.coef_, clf2.coef_, 0.2)
