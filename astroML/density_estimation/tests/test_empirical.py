import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm
from astroML.density_estimation import\
    EmpiricalDistribution, FunctionDistribution


def test_empirical_distribution(N=1000, rseed=0):
    np.random.seed(rseed)
    X = norm.rvs(0, 1, size=N)
    dist = EmpiricalDistribution(X)
    X2 = dist.rvs(N)

    meanX = X.mean()
    meanX2 = X2.mean()

    stdX = X.std()
    stdX2 = X2.std()

    assert_allclose([meanX, stdX], [meanX2, stdX2], atol=3 / np.sqrt(N))


def test_function_distribution(N=1000, rseed=0):
    f = norm(0, 1).pdf
    # go from -10 to 10 to check interpolation in presence of zeros
    dist = FunctionDistribution(f, -10, 10)

    np.random.seed(rseed)
    X = dist.rvs(N)

    meanX = X.mean()
    stdX = X.std()

    assert_allclose([meanX, stdX], [0, 1], atol=3 / np.sqrt(N))
