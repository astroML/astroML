import numpy as np

from numpy.testing import assert_almost_equal, assert_array_almost_equal
from astroML.utils import logsumexp, log_multivariate_gaussian, convert_2D_cov


def positive_definite_matrix(N, M=None):
    """return an array of M positive-definite matrices with shape (N, N)"""
    if M is None:
        V = np.random.random((N, N))
        V = np.dot(V, V.T)
    else:
        V = np.random.random((M, N, N))
        for i in range(M):
            V[i] = np.dot(V[i], V[i].T)
    return V


def test_logsumexp():
    np.random.seed(0)
    X = np.random.random((100, 100))

    for axis in (None, 0, 1):
        np_result = np.log(np.sum(np.exp(X), axis=axis))
        aML_result = logsumexp(X, axis=axis)

        assert_array_almost_equal(np_result, aML_result)


def test_log_multivariate_gaussian_methods():
    np.random.seed(0)
    x = np.random.random(3)
    mu = np.random.random(3)
    V = positive_definite_matrix(3, M=10)

    res1 = log_multivariate_gaussian(x, mu, V, method=0)
    res2 = log_multivariate_gaussian(x, mu, V, method=1)

    assert_array_almost_equal(res1, res2)


def test_log_multivariate_gaussian():
    np.random.seed(0)
    x = np.random.random((2, 1, 1, 3))
    mu = np.random.random((3, 1, 3))

    V = positive_definite_matrix(3, M=4)

    res1 = log_multivariate_gaussian(x, mu, V)
    assert res1.shape == (2, 3, 4)

    res2 = np.zeros_like(res1)
    for i in range(2):
        for j in range(3):
            for k in range(4):
                res2[i, j, k] = log_multivariate_gaussian(x[i, 0, 0],
                                                          mu[j, 0],
                                                          V[k])
    assert_array_almost_equal(res1, res2)


def test_log_multivariate_gaussian_Vinv():
    np.random.seed(0)
    x = np.random.random((2, 1, 1, 3))
    mu = np.random.random((3, 1, 3))

    V = positive_definite_matrix(3, M=4)
    Vinv = np.array([np.linalg.inv(Vi) for Vi in V])

    res1 = log_multivariate_gaussian(x, mu, V)
    res2 = log_multivariate_gaussian(x, mu, V, Vinv=Vinv)

    assert_array_almost_equal(res1, res2)


def test_2D_cov():
    s1 = 1.3
    s2 = 1.0
    alpha = 0.2

    cov = convert_2D_cov(s1, s2, alpha)
    assert_array_almost_equal([s1, s2, alpha],
                              convert_2D_cov(cov))
