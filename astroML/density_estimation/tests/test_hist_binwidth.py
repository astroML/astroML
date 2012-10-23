import numpy as np
from numpy.testing import assert_allclose, assert_
from astroML.density_estimation import \
    scotts_bin_width, freedman_bin_width, knuth_bin_width, histogram


def test_scotts_bin_width(N=10000, rseed=0):
    np.random.seed(rseed)
    X = np.random.normal(size=N)
    delta = scotts_bin_width(X)

    assert_allclose(delta,  3.5 * np.std(X) / N ** (1. / 3))


def test_freedman_bin_width(N=10000, rseed=0):
    np.random.seed(rseed)
    X = np.random.normal(size=N)
    delta = freedman_bin_width(X)

    indices = np.argsort(X)
    i25 = indices[N / 4 - 1]
    i75 = indices[(3 * N) / 4 - 1]

    assert_allclose(delta, 2 * (X[i75] - X[i25]) / N ** (1. / 3))


def test_knuth_bin_width(N=10000, rseed=0):
    np.random.seed(0)
    X = np.random.normal(size=N)
    dx, bins = knuth_bin_width(X, return_bins=True)
    assert_allclose(len(bins), 59)


def test_histogram(N=1000, rseed=0):
    np.random.seed(0)
    x = np.random.normal(0, 1, N)

    for bins in [30, np.linspace(-5, 5, 31),
                 'knuth', 'scotts', 'freedman']:
        counts, bins = histogram(x, bins)
        assert_(counts.sum() == len(x))
        assert_(len(counts) == len(bins) - 1)
