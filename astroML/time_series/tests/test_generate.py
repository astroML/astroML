import numpy as np
from numpy.testing import assert_, assert_almost_equal
from astroML.time_series import generate_power_law, generate_damped_RW


def check_generate_args(N, dt, beta, generate_complex):
    x = generate_power_law(N, dt, beta, generate_complex)

    assert_(bool(generate_complex) == np.iscomplexobj(x))
    assert_(len(x) == N)


def test_generate_args():
    dt = 0.1
    beta = 2
    for N in [10, 11]:
        for generate_complex in [True, False]:
            yield (check_generate_args, N, dt, beta, generate_complex)


def test_generate_RW():
    t = np.arange(0., 1E2)
    tau = 300
    z = 2.0
    xmean = np.random.random(1)*200 - 100
    N = len(t)
    y = generate_damped_RW(t, tau=tau, z=z, xmean=xmean)

    assert_(len(generate_damped_RW(t)) == N)
    assert_almost_equal(np.mean(y), xmean, 0)
