import numpy as np

from numpy.testing import assert_array_almost_equal, assert_allclose
from astroML.utils import log_multivariate_gaussian, convert_2D_cov, completeness_contamination, split_samples


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


def test_completeness_contamination():
    completeness, contamination = \
                    completeness_contamination(np.ones(100), np.ones(100))

    assert_allclose(completeness, 1)
    assert_allclose(contamination, 0)

    completeness, contamination = \
                    completeness_contamination(np.zeros(100), np.zeros(100))

    assert_allclose(completeness, 0)
    assert_allclose(contamination, 0)

    completeness, contamination = \
                completeness_contamination(
                    np.concatenate((np.ones(50), np.zeros(50))),
                    np.concatenate((np.ones(25), np.zeros(50), np.ones(25)))
                )

    assert_allclose(completeness, 0.5)
    assert_allclose(contamination, 0.5)


def test_split_samples():
    X = np.arange(100.)
    y = np.arange(100.)

    X_divisions, y_divisions = split_samples(X, y)

    assert (len(X_divisions[0]) == len(y_divisions[0]) == 75)
    assert (len(X_divisions[1]) == len(y_divisions[1]) == 25)
    assert (len(set(X_divisions[0]) | set(X_divisions[1])) == 100)
