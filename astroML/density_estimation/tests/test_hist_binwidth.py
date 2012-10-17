import numpy as np
from astroML.density_estimation import \
    scotts_bin_width, freedman_bin_width, knuth_nbins

def test_scotts_bin_width(N=10000, rseed=0):
    np.random.seed(rseed)
    X = np.random.normal(size=N)
    delta = scotts_bin_width(X)

    assert delta == 3.5 * np.std(X) / N ** (1. / 3)


def test_freedman_bin_width(N=10000, rseed=0):
    np.random.seed(rseed)
    X = np.random.normal(size=N)
    delta = freedman_bin_width(X)

    indices = np.argsort(X)
    i25 = indices[N/4 - 1]
    i75 = indices[(3 * N) / 4 - 1]
    
    assert delta == 2 * (X[i75] - X[i25]) / N ** (1./ 3)


def test_knuth_nbins(N=10000, rseed=0):
    np.random.seed(0)
    X = np.random.normal(size=N)
    nbins = knuth_nbins(X)
    assert nbins == 56
