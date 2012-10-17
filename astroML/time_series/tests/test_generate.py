import numpy as np
from numpy.testing import assert_
from astroML.time_series import generate_power_law


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
