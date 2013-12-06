import numpy as np
from numpy.testing import assert_allclose
from astroML.filters import savitzky_golay, wiener_filter


def test_savitzky_golay():
    y = np.zeros(100)
    y[::2] = 1
    f = savitzky_golay(y, window_size=3, order=1)
    assert_allclose(f, (2 - y) / 3.)


def test_savitzky_golay_fft():
    y = np.random.normal(size=100)

    for width in [3, 5]:
        for order in range(width - 1):
            f1 = savitzky_golay(y, width, order, use_fft=False)
            f2 = savitzky_golay(y, width, order, use_fft=True)
            assert_allclose(f1, f2)


def test_wiener_filter_simple():
    t = np.linspace(0, 1, 256)
    h = np.zeros_like(t)
    h[::2] = 1000
    s = wiener_filter(t, h)
    assert_allclose(s, np.mean(h))


def test_wienter_filter_spike():
    np.random.seed(0)
    N = 2048
    dt = 0.05

    t = dt * np.arange(N)
    h = np.exp(-0.5 * ((t - 20.) / 1.0) ** 2) + 10
    hN = h + np.random.normal(0, 0.05, size=h.shape)
    h_smooth = wiener_filter(t, hN)

    assert_allclose(h, h_smooth, atol=0.03)
