import numpy as np
from numpy.testing import assert_allclose
from astroML.fourier import\
    FT_continuous, IFT_continuous, PSD_continuous, sinegauss, sinegauss_FT


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


def sinegauss(t, t0, f0, a):
    """Sine-gaussian wavelet"""
    return (np.exp(-a * (t - t0) ** 2)
            * np.exp(2j * np.pi * f0 * (t - t0)))


def sinegauss_FT(f, t0, f0, a):
    """Fourier transform of the sine-gaussian wavelet.
    This uses the convention H(f) = integral[ h(t) exp(-2pi i f t) dt]
    """
    return (np.sqrt(np.pi / a)
            * np.exp(-2j * np.pi * f * t0)
            * np.exp(-np.pi ** 2 * (f - f0) ** 2 / a))


def sinegauss_PSD(f, t0, f0, a):
    """PSD of the sine-gaussian wavelet
    PSD(f) = |H(f)|^2 + |H(-f)|^2
    """
    Pf = np.pi / a * np.exp(-2 * np.pi ** 2 * (f - f0) ** 2 / a)
    Pmf = np.pi / a * np.exp(-2 * np.pi ** 2 * (-f - f0) ** 2 / a)
    return Pf + Pmf


def check_FT_continuous(a, t0, f0, method, t):
    h = sinegauss(t, t0, f0, a)
    f, H = FT_continuous(t, h, method=method)
    assert_allclose(H, sinegauss_FT(f, t0, f0, a), atol=1E-12)


def test_FT_continuous():
    t = np.linspace(-9, 10, 10000)
    for a in (1, 2):
        for t0 in (-2, 0, 2):
            for f0 in (-1, 0, 1):
                for method in (1, 2):
                    yield (check_FT_continuous, a, t0, f0, method, t)


def check_PSD_continuous(a, t0, f0, method, t):
    h = sinegauss(t, t0, f0, a)
    f, P = PSD_continuous(t, h, method=method)
    assert_allclose(P, sinegauss_PSD(f, t0, f0, a), atol=1E-12)


def test_PSD_continuous():
    t = np.linspace(-9, 10, 10000)
    for a in (1, 2):
        for t0 in (-2, 0, 2):
            for f0 in (-1, 0, 1):
                for method in (1, 2):
                    yield (check_PSD_continuous, a, t0, f0, method, t)


def check_IFT_continuous(a, t0, f0, method, f):
    H = sinegauss_FT(f, t0, f0, a)
    t, h = IFT_continuous(f, H, method=method)
    assert_allclose(h, sinegauss(t, t0, f0, a), atol=1E-12)


def test_IFT_continuous():
    f = np.linspace(-9, 10, 10000)
    for a in (1, 2):
        for t0 in (-2, 0, 2):
            for f0 in (-1, 0, 1):
                for method in (1, 2):
                    yield (check_IFT_continuous, a, t0, f0, method, f)


def test_IFT_FT():
    # Test IFT(FT(x)) = x
    np.random.seed(0)
    t = -50 + 0.01 * np.arange(10000.)
    x = np.random.random(10000)

    f, y = FT_continuous(t, x)
    t, xp = IFT_continuous(f, y)

    assert_allclose(x, xp, atol=1E-7)
