import pytest
import numpy as np
from numpy.testing import assert_allclose
from astroML.fourier import\
    FT_continuous, IFT_continuous, PSD_continuous, sinegauss, sinegauss_FT


@pytest.mark.parametrize('t0', [-1, 0, 1])
@pytest.mark.parametrize('f0', [1, 2])
@pytest.mark.parametrize('Q', [1, 2])
def test_wavelets(t0, f0, Q):
    t = np.linspace(-10, 10, 10000)
    h = sinegauss(t, t0, f0, Q)
    f, H = FT_continuous(t, h)
    H2 = sinegauss_FT(f, t0, f0, Q)
    assert_allclose(H, H2, atol=1E-8)


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


@pytest.mark.parametrize('a', [1, 2])
@pytest.mark.parametrize('t0', [-2, 0, 2])
@pytest.mark.parametrize('f0', [-1, 0, 1])
@pytest.mark.parametrize('method', [1, 2])
def test_FT_continuous(a, t0, f0, method):
    t = np.linspace(-9, 10, 10000)
    h = sinegauss(t, t0, f0, a)
    f, H = FT_continuous(t, h, method=method)
    assert_allclose(H, sinegauss_FT(f, t0, f0, a), atol=1E-12)


@pytest.mark.parametrize('a', [1, 2])
@pytest.mark.parametrize('t0', [-2, 0, 2])
@pytest.mark.parametrize('f0', [-1, 0, 1])
@pytest.mark.parametrize('method', [1, 2])
def test_PSD_continuous(a, t0, f0, method):
    t = np.linspace(-9, 10, 10000)
    h = sinegauss(t, t0, f0, a)
    f, P = PSD_continuous(t, h, method=method)
    assert_allclose(P, sinegauss_PSD(f, t0, f0, a), atol=1E-12)


@pytest.mark.parametrize('a', [1, 2])
@pytest.mark.parametrize('t0', [-2, 0, 2])
@pytest.mark.parametrize('f0', [-1, 0, 1])
@pytest.mark.parametrize('method', [1, 2])
def check_IFT_continuous(a, t0, f0, method):
    f = np.linspace(-9, 10, 10000)
    H = sinegauss_FT(f, t0, f0, a)
    t, h = IFT_continuous(f, H, method=method)
    assert_allclose(h, sinegauss(t, t0, f0, a), atol=1E-12)


def test_IFT_FT():
    # Test IFT(FT(x)) = x
    np.random.seed(0)
    t = -50 + 0.01 * np.arange(10000.)
    x = np.random.random(10000)

    f, y = FT_continuous(t, x)
    t, xp = IFT_continuous(f, y)

    assert_allclose(x, xp, atol=1E-7)
