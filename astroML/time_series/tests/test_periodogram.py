import numpy as np
from numpy.testing import assert_almost_equal
from astroML.time_series import search_frequencies


# TODO: add tests of lomb_scargle inputs & significance

# TODO: add tests of bootstrap


def test_search_frequencies():
    rng = np.random.RandomState(0)

    t = np.arange(0, 1E1, 0.01)
    f = 1
    w = 2 * np.pi * np.array(f)
    y = np.sin(w * t)

    dy = 0.01
    y += dy * rng.randn(len(y))

    omegas, power = search_frequencies(t, y, dy)
    omax = omegas[power == max(power)]

    assert_almost_equal(w, omax, decimal=3)
