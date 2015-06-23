import numpy as np
from numpy.testing import assert_, assert_almost_equal
from astroML.time_series import lomb_scargle_bootstrap
from astroML.utils import check_random_state
from astroML_addons.periodogram import lomb_scargle


def test_periodogram():
    t = np.arange(0, 10, 0.1)
    N = len(t)
    f = 1
    w = 2*np.pi*f
    omega = np.arange(0.5, 2.5, 0.001)*w
    M = len(omega)
    y = np.sin(w*t)
    dy = np.zeros(N) + 0.01
    iterations = 50
    seed = 5

    rdm = check_random_state(seed)
    all_peaks = np.zeros((iterations, M))

    for ii in range(iterations):
        ind = rdm.randint(0, N, N)
        p = lomb_scargle(t, y[ind], dy[ind], omega,
                         generalized=True, subtract_mean=True)
        all_peaks[ii] = p

    D = lomb_scargle_bootstrap(t, y, dy, omega,
                               N_bootstraps=iterations, random_state=seed)

    for ii in range(len(D)):
        assert_(D[ii] == np.max(all_peaks[ii]))
