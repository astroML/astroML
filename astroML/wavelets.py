import numpy as np
from .fourier import FT_continuous, IFT_continuous

def sinegauss(t, t0, f0, Q):
    """Sine-gaussian wavelet"""
    a = (f0 * 1. / Q) ** 2
    return (np.exp(-a * (t - t0) ** 2)
            * np.exp(2j * np.pi * f0 * (t - t0)))


def sinegauss_FT(f, t0, f0, Q):
    """Fourier transform of the sine-gaussian wavelet.

    This uses the convention
    H(f) = integral[ h(t) exp(-2pi i f t) dt]
    """
    a = (f0 * 1. / Q) ** 2
    return (np.sqrt(np.pi / a)
            * np.exp(-2j * np.pi * f * t0)
            * np.exp(-np.pi ** 2 * (f - f0) ** 2 / a))


def sinegauss_PSD(f, t0, f0, Q):
    """Compute the PSD of the sine-gaussian function at frequency f

    PSD(f) = |H(f)|^2 + |H(-f)|^2
    """
    a = (f0 * 1. / Q) ** 2
    Pf = np.pi / a * np.exp(-2 * np.pi ** 2 * (f - f0) ** 2 / a)
    Pmf = np.pi / a * np.exp(-2 * np.pi ** 2 * (-f - f0) ** 2 / a)
    return Pf + Pmf


def wavelet_PSD(t, h, f0, Q=1.0):
    """Compute the wavelet PSD as a function of f0 and t

    Parameters
    ----------
    t : array_like
        array of times, length N
    h : array_like
        array of observed values, length N
    f0 : array_like
        array of candidate frequencies, length Nf
    Q : float
        Q-parameter for wavelet

    Returns
    -------
    PSD : ndarray
        The 2-dimensional PSD, of shape (Nf, N), corresponding with
        frequencies f0 and times t.
    """
    t, h, f0 = map(np.asarray, (t, h, f0))
    if (t.ndim != 1) or (t.shape != h.shape):
        raise ValueError('t and h must be one dimensional and the same shape')

    if f0.ndim != 1:
        raise ValueError('f0 must be one dimensional')

    Q = Q + np.zeros_like(f0)

    f, H = FT_continuous(t, h)
    W = np.conj(sinegauss_FT(f, 0, f0[:, None], Q[:, None]))
    _, HW = IFT_continuous(f, H * W)

    return abs(HW) ** 2
