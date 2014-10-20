import numpy as np
from numpy.testing import assert_allclose

from ..TLS import TLS_logL


def test_TLS_likelihood_diagonal(rseed=0):
    """Test Total-Least-Squares fit with diagonal covariance"""
    np.random.seed(rseed)

    X = np.random.rand(10, 2)
    dX1 = 0.1 * np.ones((10, 2))
    dX2 = 0.1 * np.array([np.eye(2) for i in range(10)])
    v = np.random.random(2)

    assert_allclose(TLS_logL(v, X, dX1),
                    TLS_logL(v, X, dX2))
