import numpy as np

try:
    # use scipy if available: it's faster
    from scipy.fftpack import fft, ifft, fftshift
except ImportError:
    from numpy.fft import fft, ifft, fftshift


def FT_continuous(t, h, axis=-1, method=1):
    r"""Approximate a continuous 1D Fourier Transform with sampled data.

    This function uses the Fast Fourier Transform to approximate
    the continuous fourier transform of a sampled function, using
    the convention

    .. math::

       H(f) = \int h(t) exp(-2 \pi i f t) dt

    It returns f and H, which approximate H(f).

    Parameters
    ----------
    t : array_like
        regularly sampled array of times
        t is assumed to be regularly spaced, i.e.
        t = t0 + Dt * np.arange(N)
    h : array_like
        real or complex signal at each time
    axis : int
        axis along which to perform fourier transform.
        This axis must be the same length as t.

    Returns
    -------
    f : ndarray
        frequencies of result.  Units are the same as 1/t
    H : ndarray
        Fourier coefficients at each frequency.
    """
    assert t.ndim == 1
    assert h.shape[axis] == t.shape[0]
    N = len(t)
    if N % 2 != 0:
        raise ValueError("number of samples must be even")

    Dt = t[1] - t[0]
    Df = 1. / (N * Dt)
    t0 = t[N // 2]

    f = Df * (np.arange(N) - N // 2)

    shape = np.ones(h.ndim, dtype=int)
    shape[axis] = N

    phase = np.ones(N)
    phase[1::2] = -1
    phase = phase.reshape(shape)

    if method == 1:
        H = Dt * fft(h * phase, axis=axis)
    else:
        H = Dt * fftshift(fft(h, axis=axis), axes=axis)

    H *= phase
    H *= np.exp(-2j * np.pi * t0 * f.reshape(shape))
    H *= np.exp(-1j * np.pi * N / 2)

    return f, H


def IFT_continuous(f, H, axis=-1, method=1):
    """Approximate a continuous 1D Inverse Fourier Transform with sampled data.

    This function uses the Fast Fourier Transform to approximate
    the continuous fourier transform of a sampled function, using
    the convention

    .. math::

       H(f) = integral[ h(t) exp(-2 pi i f t) dt]

       h(t) = integral[ H(f) exp(2 pi i f t) dt]

    It returns t and h, which approximate h(t).

    Parameters
    ----------
    f : array_like
        regularly sampled array of times
        t is assumed to be regularly spaced, i.e.
        f = f0 + Df * np.arange(N)
    H : array_like
        real or complex signal at each time
    axis : int
        axis along which to perform fourier transform.
        This axis must be the same length as t.

    Returns
    -------
    f : ndarray
        frequencies of result.  Units are the same as 1/t
    H : ndarray
        Fourier coefficients at each frequency.
    """
    assert f.ndim == 1
    assert H.shape[axis] == f.shape[0]
    N = len(f)
    if N % 2 != 0:
        raise ValueError("number of samples must be even")

    f0 = f[0]
    Df = f[1] - f[0]

    t0 = -0.5 / Df
    Dt = 1. / (N * Df)
    t = t0 + Dt * np.arange(N)

    shape = np.ones(H.ndim, dtype=int)
    shape[axis] = N

    t_calc = t.reshape(shape)
    f_calc = f.reshape(shape)

    H_prime = H * np.exp(2j * np.pi * t0 * f_calc)
    h_prime = ifft(H_prime, axis=axis)
    h = N * Df * np.exp(2j * np.pi * f0 * (t_calc - t0)) * h_prime

    return t, h


def PSD_continuous(t, h, axis=-1, method=1):
    r"""Approximate a continuous 1D Power Spectral Density of sampled data.

    This function uses the Fast Fourier Transform to approximate
    the continuous fourier transform of a sampled function, using
    the convention

    .. math::

        H(f) = \int h(t) \exp(-2 \pi i f t) dt

    It returns f and PSD, which approximate PSD(f) where

    .. math::

        PSD(f) = |H(f)|^2 + |H(-f)|^2

    Parameters
    ----------
    t : array_like
        regularly sampled array of times
        t is assumed to be regularly spaced, i.e.
        t = t0 + Dt * np.arange(N)
    h : array_like
        real or complex signal at each time
    axis : int
        axis along which to perform fourier transform.
        This axis must be the same length as t.

    Returns
    -------
    f : ndarray
        frequencies of result.  Units are the same as 1/t
    PSD : ndarray
        Fourier coefficients at each frequency.
    """
    assert t.ndim == 1
    assert h.shape[axis] == t.shape[0]
    N = len(t)
    if N % 2 != 0:
        raise ValueError("number of samples must be even")

    ax = axis % h.ndim

    if method == 1:
        # use FT_continuous
        f, Hf = FT_continuous(t, h, axis)
        Hf = np.rollaxis(Hf, ax)
        f = -f[N // 2::-1]
        PSD = abs(Hf[N // 2::-1]) ** 2
        PSD[:-1] += abs(Hf[N // 2:]) ** 2
        PSD = np.rollaxis(PSD, 0, ax + 1)
    else:
        # A faster way to do it is with fftshift
        # take advantage of the fact that phases go away
        Dt = t[1] - t[0]
        Df = 1. / (N * Dt)
        f = Df * np.arange(N // 2 + 1)
        Hf = fft(h, axis=axis)
        Hf = np.rollaxis(Hf, ax)
        PSD = abs(Hf[:N // 2 + 1]) ** 2
        PSD[-1] = 0
        PSD[1:] += abs(Hf[N // 2:][::-1]) ** 2
        PSD[0] *= 2
        PSD = Dt ** 2 * np.rollaxis(PSD, 0, ax + 1)

    return f, PSD


def sinegauss(t, t0, f0, Q):
    """Sine-gaussian wavelet"""
    a = (f0 * 1. / Q) ** 2
    return (np.exp(-a * (t - t0) ** 2)
            * np.exp(2j * np.pi * f0 * (t - t0)))


def sinegauss_FT(f, t0, f0, Q):
    """Fourier transform of the sine-gaussian wavelet.

    This uses the convention

    .. math::

       H(f) = integral[ h(t) exp(-2pi i f t) dt]
    """
    a = (f0 * 1. / Q) ** 2
    return (np.sqrt(np.pi / a)
            * np.exp(-2j * np.pi * f * t0)
            * np.exp(-np.pi ** 2 * (f - f0) ** 2 / a))


def sinegauss_PSD(f, t0, f0, Q):
    """Compute the PSD of the sine-gaussian function at frequency f

    .. math::

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
