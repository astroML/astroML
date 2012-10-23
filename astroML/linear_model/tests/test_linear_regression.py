import numpy as np
from numpy.testing import assert_allclose

from sklearn.linear_model import LinearRegression as skLinearRegression
from astroML.linear_model import \
    LinearRegression, PolynomialRegression, BasisFunctionRegression


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
