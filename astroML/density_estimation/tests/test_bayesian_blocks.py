import numpy as np
from astroML.density_estimation.bayesian_blocks import _find_change_point

def test_find_change_point():
    # make sure the correct change point is found
    # for a simple distribution
    nphot = 3000
    nphot_unit = int(nphot / 3.)
    nticks = 10 * nphot_unit

    np.random.seed(0)
    first_batch = nticks * np.random.random(nphot_unit)
    second_batch = nticks * (1 + np.random.random(2 * nphot_unit))
    photon_times = np.concatenate([first_batch, second_batch])
    photon_times.sort()

    delta_t = (photon_times[-1] - photon_times[0]) / (photon_times.size - 1.)
    t_0 = photon_times[0] - delta_t
    t_1 = photon_times[-1] - delta_t

    (change_point, odds_21,
     log_prob, log_prob_noseg) = _find_change_point(photon_times, t_0, t_1)

    assert abs(change_point - nphot_unit) < 0.01 * nphot_unit
    assert log_prob[change_point] == np.max(log_prob)
