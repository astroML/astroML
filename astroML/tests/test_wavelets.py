import numpy as np
from numpy.testing import assert_allclose
from astroML.wavelets import sinegauss, sinegauss_FT
from astroML.fourier import FT_continuous

def check_wavelets(t0, f0, Q, t):
    h = sinegauss(t, t0, f0, Q)
    f, H = FT_continuous(t, h)
    H2 = sinegauss_FT(f, t0, f0, Q)
    assert_allclose(H, H2, atol=1E-8)
    

def test_wavelets():
    t = np.linspace(-10, 10, 10000)
    for t0 in (-1, 0, 1):
        for f0 in (1, 2):
            for Q in (1, 2):
                yield (check_wavelets, t0, f0, Q, t)
