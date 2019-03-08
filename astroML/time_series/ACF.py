"""
Auto-correlation functions
"""
import numpy as np
from scipy import fftpack
from .periodogram import lomb_scargle


def ACF_scargle(t, y, dy, n_omega=2 ** 10, omega_max=100):
    """Compute the Auto-correlation function via Scargle's method

    Parameters
    ----------
    t : array_like
        times of observation.  Assumed to be in increasing order.
    y : array_like
        values of each observation.  Should be same shape as t
    dy : float or array_like
        errors in each observation.
    n_omega : int (optional)
        number of angular frequencies at which to evaluate the periodogram
        default is 2^10
    omega_max : float (optional)
        maximum value of omega at which to evaluate the periodogram
        default is 100

    Returns
    -------
    ACF, t : ndarrays
        The auto-correlation function and associated times
    """
    t = np.asarray(t)
    y = np.asarray(y)

    if y.shape != t.shape:
        raise ValueError("shapes of t and y must match")

    dy = np.asarray(dy) * np.ones(y.shape)

    d_omega = omega_max * 1. / (n_omega + 1)
    omega = d_omega * np.arange(1, n_omega + 1)

    # recall that P(omega = 0) = (chi^2(0) - chi^2(0)) / chi^2(0)
    #                          = 0

    # compute P and shifted full-frequency array
    P = lomb_scargle(t, y, dy, omega,
                     generalized=True)
    P = np.concatenate([[0], P, P[-2::-1]])

    # compute PW, the power of the window function
    PW = lomb_scargle(t, np.ones(len(t)), dy, omega,
                      generalized=False, subtract_mean=False)
    PW = np.concatenate([[0], PW, PW[-2::-1]])

    # compute the  inverse fourier transform of P and PW
    rho = fftpack.ifft(P).real
    rhoW = fftpack.ifft(PW).real

    ACF = fftpack.fftshift(rho / rhoW) / np.sqrt(2)
    N = len(ACF)
    dt = 2 * np.pi / N / (omega[1] - omega[0])
    t = dt * (np.arange(N) - N // 2)

    return ACF, t


def ACF_EK(t, y, dy, bins=20):
    """Auto-correlation function via the Edelson-Krolik method

    Parameters
    ----------
    t : array_like
        times of observation.  Assumed to be in increasing order.
    y : array_like
        values of each observation.  Should be same shape as t
    dy : float or array_like
        errors in each observation.
    bins : int or array_like (optional)
        if integer, the number of bins to use in the analysis.
        if array, the (nbins + 1) bin edges.
        Default is bins=20.

    Returns
    -------
    ACF : ndarray
        The auto-correlation function and associated times
    err : ndarray
        the error in the ACF
    bins : ndarray
        bin edges used in computation
    """
    t = np.asarray(t)
    y = np.asarray(y)

    if y.shape != t.shape:
        raise ValueError("shapes of t and y must match")

    if t.ndim != 1:
        raise ValueError("t should be a 1-dimensional array")

    dy = np.asarray(dy) * np.ones(y.shape)

    # compute mean and standard deviation of y
    w = 1. / dy / dy
    w /= w.sum()
    mu = np.dot(w, y)
    sigma = np.std(y, ddof=1)

    dy2 = dy[:, None]

    dt = t - t[:, None]
    UDCF = ((y - mu) * (y - mu)[:, None] /
            np.sqrt((sigma ** 2 - dy ** 2) *
                    (sigma ** 2 - dy2 ** 2)))

    # determine binning
    bins = np.asarray(bins)
    if bins.size == 1:
        dt_min = dt.min()
        dt_max = dt.max()
        bins = np.linspace(dt_min, dt_max + 1E-10, bins + 1)

    ACF = np.zeros(len(bins) - 1)
    M = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        flag = (dt >= bins[i]) & (dt < bins[i + 1])
        M[i] = flag.sum()
        ACF[i] = np.sum(UDCF[flag])

    ACF /= M

    return ACF, np.sqrt(2. / M), bins
