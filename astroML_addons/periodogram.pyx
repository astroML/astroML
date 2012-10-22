import numpy as np
cimport numpy as np

cimport cython

cdef extern from "math.h":
    double sin(double)
    double cos(double)
    double atan(double)

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

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
        computing the periodogram.  Only referenced if generalized is False.
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
    t = np.asarray(t, dtype=DTYPE, order='C')
    y = np.asarray(y, dtype=DTYPE, order='C')
    dy = np.asarray(dy) * np.ones_like(y)
    omega = np.asarray(omega, dtype=DTYPE, order='C')

    assert t.ndim == 1
    assert y.ndim == 1
    assert dy.ndim == 1
    assert omega.ndim == 1
    assert t.shape == y.shape
    assert y.shape == dy.shape

    p_omega = np.zeros(omega.shape, dtype=DTYPE, order='C')

    # the generalized method takes care of offset automatically,
    # while the classic method requires centered data.
    if (not generalized) and subtract_mean:
        # compute MLE for mean in the presence of noise.
        w = 1. / dy ** 2
        y = y - np.dot(w, y) / np.sum(w)
    
    if generalized:
        _generalized_lomb_scargle(t, y, dy, omega, p_omega)
    else:
        _standard_lomb_scargle(t, y, dy, omega, p_omega)

    if significance is not None:
        N = t.size
        M = 2 * N
        z = (-2.0 / (N - 1.)
             * np.log(1 - (1 - np.asarray(significance)) ** (1. / M)))
        return p_omega, z
    else:
        return p_omega


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _standard_lomb_scargle(np.ndarray[DTYPE_t, ndim=1, mode='c'] t,
                            np.ndarray[DTYPE_t, ndim=1, mode='c'] y,
                            np.ndarray[DTYPE_t, ndim=1, mode='c'] dy,
                            np.ndarray[DTYPE_t, ndim=1, mode='c'] omega,
                            np.ndarray[DTYPE_t, ndim=1, mode='c'] p_omega):
    cdef ITYPE_t N_freq = omega.shape[0]
    cdef ITYPE_t N_obs = t.shape[0]

    cdef unsigned int i, j
    cdef DTYPE_t w, omega_t, sin_omega_t, cos_omega_t, tan_2omega_tau
    cdef DTYPE_t S2, C2, tau, Y, wsum, YY, YCtau, YStau, CCtau, SStau

    for i from 0 <= i < N_freq:
        # first pass: determine tau
        S2 = 0
        C2 = 0
        for j from 0 <= j < N_obs:
            w = 1. / dy[j]
            w *= w
        
            omega_t = omega[i] * t[j]
            sin_omega_t = sin(omega_t)
            cos_omega_t = cos(omega_t)

            S2 += 2 * w * sin_omega_t * cos_omega_t
            C2 += w - 2 * w * sin_omega_t * sin_omega_t

        tan_2omega_tau = S2 / C2
        tau = atan(tan_2omega_tau)
        tau *= 0.5
        tau /= omega[i]

        wsum = 0
        Y = 0
        YY = 0
        YCtau = 0
        YStau = 0
        CCtau = 0
        SStau = 0

        # second pass: compute the power
        for j from 0 <= j < N_obs:
            w = 1. / dy[j]
            w *= w
            wsum += w

            omega_t = omega[i] * (t[j] - tau)
            sin_omega_t = sin(omega_t)
            cos_omega_t = cos(omega_t)

            Y += w * y[j]
            YY += w * y[j] * y[j]
            YCtau += w * y[j] * cos_omega_t
            YStau += w * y[j] * sin_omega_t
            CCtau += w * cos_omega_t * cos_omega_t
            SStau += w * sin_omega_t * sin_omega_t

        Y /= wsum
        YY /= wsum
        YCtau /= wsum
        YStau /= wsum
        CCtau /= wsum
        SStau /= wsum
        
        p_omega[i] = (YCtau * YCtau / CCtau + YStau * YStau / SStau) / YY


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _generalized_lomb_scargle(np.ndarray[DTYPE_t, ndim=1, mode='c'] t,
                               np.ndarray[DTYPE_t, ndim=1, mode='c'] y,
                               np.ndarray[DTYPE_t, ndim=1, mode='c'] dy,
                               np.ndarray[DTYPE_t, ndim=1, mode='c'] omega,
                               np.ndarray[DTYPE_t, ndim=1, mode='c'] p_omega):
    cdef ITYPE_t N_freq = omega.shape[0]
    cdef ITYPE_t N_obs = t.shape[0]

    cdef unsigned int i, j
    cdef DTYPE_t w, omega_t, sin_omega_t, cos_omega_t, tan_2omega_tau
    cdef DTYPE_t S, C, S2, C2, tau, Y, wsum, YY
    cdef DTYPE_t Stau, Ctau, YCtau, YStau, CCtau, SStau

    for i from 0 <= i < N_freq:
        # first pass: determine tau
        wsum = 0
        S = 0
        C = 0
        S2 = 0
        C2 = 0
        for j from 0 <= j < N_obs:
            w = 1. / dy[j]
            w *= w
            wsum += w
        
            omega_t = omega[i] * t[j]
            sin_omega_t = sin(omega_t)
            cos_omega_t = cos(omega_t)

            S += w * sin_omega_t
            C += w * cos_omega_t

            S2 += 2 * w * sin_omega_t * cos_omega_t
            C2 += w - 2 * w * sin_omega_t * sin_omega_t

        S2 /= wsum
        C2 /= wsum
        S /= wsum
        C /= wsum
            
        S2 -= (2 * S * C)
        C2 -= (C * C - S * S)

        tan_2omega_tau = S2 / C2
        tau = atan(tan_2omega_tau)
        tau *= 0.5
        tau /= omega[i]

        Y = 0
        YY = 0
        Stau = 0
        Ctau = 0
        YCtau = 0
        YStau = 0
        CCtau = 0
        SStau = 0

        # second pass: compute the power
        for j from 0 <= j < N_obs:
            w = 1. / dy[j]
            w *= w

            omega_t = omega[i] * (t[j] - tau)
            sin_omega_t = sin(omega_t)
            cos_omega_t = cos(omega_t)

            Y += w * y[j]
            YY += w * y[j] * y[j]
            Ctau += w * cos_omega_t
            Stau += w * sin_omega_t
            YCtau += w * y[j] * cos_omega_t
            YStau += w * y[j] * sin_omega_t
            CCtau += w * cos_omega_t * cos_omega_t
            SStau += w * sin_omega_t * sin_omega_t

        Y /= wsum
        YY /= wsum
        Ctau /= wsum
        Stau /= wsum
        YCtau /= wsum
        YStau /= wsum
        CCtau /= wsum
        SStau /= wsum

        YCtau -= Y * Ctau
        YStau -= Y * Stau
        CCtau -= Ctau * Ctau
        SStau -= Stau * Stau
            
        YY -= Y * Y

        p_omega[i] = (YCtau * YCtau / CCtau + YStau * YStau / SStau) / YY
