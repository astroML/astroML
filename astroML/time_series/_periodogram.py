import numpy as np


def lomb_scargle(t, y, dy, omega, generalized=True,
                 subtract_mean=True, significance=None):
    """
    (Generalized) Lomb-Scargle Periodogram with Floating Mean

    Parameters
    ----------
    t : array_like
        sequence of times
    y : array_like
        sequence of observations
    dy : array_like
        sequence of observational errors
    omega : array_like
        frequencies at which to evaluate p(omega)
    generalized : bool
        if True (default) use generalized lomb-scargle method
        otherwise, use classic lomb-scargle.
    subtract_mean : bool
        if True (default) subtract the sample mean from the data before
        computing the periodogram.  Only referenced if generalized is False
    significance : None or float or ndarray
        if specified, then this is a list of significances to compute
        for the results.

    Returns
    -------
    p : array_like
        Lomb-Scargle power associated with each frequency omega
    z : array_like
        if significance is specified, this gives the levels corresponding
        to the desired significance (using the Scargle 1982 formalism)

    Notes
    -----
    The algorithm is based on reference [1]_.  The result for generalized=False
    is given by equation 4 of this work, while the result for generalized=True
    is given by equation 20.

    Note that the normalization used in this reference is different from that
    used in other places in the literature (e.g. [2]_).  For a discussion of
    normalization and false-alarm probability, see [1]_.

    To recover the normalization used in Scargle [3]_, the results should
    be multiplied by (N - 1) / 2 where N is the number of data points.

    References
    ----------
    .. [1] M. Zechmeister and M. Kurster, A&A 496, 577-584 (2009)
    .. [2] W. Press et al, Numerical Recipies in C (2002)
    .. [3] Scargle, J.D. 1982, ApJ 263:835-853
    """
    t = np.asarray(t)
    y = np.asarray(y)
    dy = np.asarray(dy) * np.ones_like(y)

    assert t.ndim == 1
    assert y.ndim == 1
    assert dy.ndim == 1
    assert t.shape == y.shape
    assert y.shape == dy.shape

    w = 1. / dy / dy
    w /= w.sum()

    # the generalized method takes care of offset automatically,
    # while the classic method requires centered data.
    if (not generalized) and subtract_mean:
        # subtract MLE for mean in the presence of noise.
        y = y - np.dot(w, y)

    omega = np.asarray(omega)
    shape = omega.shape
    omega = omega.ravel()[np.newaxis, :]

    t = t[:, np.newaxis]
    y = y[:, np.newaxis]
    dy = dy[:, np.newaxis]
    w = w[:, np.newaxis]

    sin_omega_t = np.sin(omega * t)
    cos_omega_t = np.cos(omega * t)

    # compute time-shift tau
    # S2 = np.dot(w.T, np.sin(2 * omega * t)
    S2 = 2 * np.dot(w.T, sin_omega_t * cos_omega_t)
    # C2 = np.dot(w.T, np.cos(2 * omega * t)
    C2 = 2 * np.dot(w.T, 0.5 - sin_omega_t ** 2)

    if generalized:
        S = np.dot(w.T, sin_omega_t)
        C = np.dot(w.T, cos_omega_t)

        S2 -= (2 * S * C)
        C2 -= (C * C - S * S)

    tan_2omega_tau = S2 / C2
    tau = np.arctan(tan_2omega_tau)
    tau *= 0.5
    tau /= omega

    # compute components needed for the fit
    omega_t_tau = omega * (t - tau)

    sin_omega_t_tau = np.sin(omega_t_tau)
    cos_omega_t_tau = np.cos(omega_t_tau)

    Y = np.dot(w.T, y)
    YY = np.dot(w.T, y * y) - Y * Y

    wy = w * y

    YCtau = np.dot(wy.T, cos_omega_t_tau)
    YStau = np.dot(wy.T, sin_omega_t_tau)
    CCtau = np.dot(w.T, cos_omega_t_tau * cos_omega_t_tau)
    SStau = np.dot(w.T, sin_omega_t_tau * sin_omega_t_tau)

    if generalized:
        Ctau = np.dot(w.T, cos_omega_t_tau)
        Stau = np.dot(w.T, sin_omega_t_tau)

        YCtau -= Y * Ctau
        YStau -= Y * Stau
        CCtau -= Ctau * Ctau
        SStau -= Stau * Stau

    p_omega = (YCtau * YCtau / CCtau + YStau * YStau / SStau) / YY
    p_omega = p_omega.reshape(shape)

    if significance is not None:
        N = t.size
        M = 2 * N
        z = (-2.0 / (N - 1.)
             * np.log(1 - (1 - np.asarray(significance)) ** (1. / M)))
        return p_omega, z
    else:
        return p_omega
