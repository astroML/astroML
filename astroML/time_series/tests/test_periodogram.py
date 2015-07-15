import numpy as np
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from astroML.time_series import lomb_scargle_bootstrap, search_frequencies
from astroML.utils import check_random_state
from astroML_addons.periodogram import lomb_scargle as lomb_scargle_addon
from astroML.time_series._periodogram \
    import lomb_scargle as lomb_scargle_native


def test_lomb_scargle():
    t = np.arange(0, 1E1, 0.01)
    f = 1
    w = 2*np.pi*np.array(f)
    y1 = np.sin(w*t)
    dy1 = np.random.normal(0, 0.1, size=len(y1))
    y1 = y1 + dy1

    y2 = np.sin(w*t)
    dy2 = np.random.normal(0, 1, size=len(y2))
    y2 = y2 + dy2

    def check_lomb_scargle(generalized, subtract_mean, significance=None):
        p1 = lomb_scargle_addon(t, y1, dy1, omegas,
                                generalized=generalized,
                                subtract_mean=subtract_mean,
                                significance=significance)
        p2 = lomb_scargle_native(t, y1, dy1, omegas,
                                 generalized=generalized,
                                 subtract_mean=subtract_mean,
                                 significance=significance)
        assert_allclose(p1, p2, rtol=5E-02)

    omegas = np.linspace(0.5, 2.5, 100)*w
    # test that for high S/N addon and native produce same results
    for generalized in [True, False]:
        for subtract_mean in [True, False]:
            yield check_lomb_scargle, generalized, subtract_mean


def test_search_frequencies():
    t = np.arange(0, 1E1, 0.01)
    f = 1
    w = 2*np.pi*np.array(f)
    y = np.sin(w*t)

    dy = np.random.normal(0, 0.1, size=len(y))
    y = y + dy

    omegas, power = search_frequencies(t, y, dy)
    omax = omegas[power == max(power)]

    assert_almost_equal(w, omax, decimal=3)
