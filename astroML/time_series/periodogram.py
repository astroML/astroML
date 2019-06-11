import numpy as np
from sklearn.utils import check_random_state

try:
    # astropy.timeseries is new in v3.2
    from astropy.timeseries import LombScargle
except ImportError:
    from astropy.stats import LombScargle

from ..utils.decorators import deprecated
from ..utils.exceptions import AstroMLDeprecationWarning


@deprecated('0.4', alternative='astropy.stats.LombScargle',
            warning_type=AstroMLDeprecationWarning)
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
    .. [2] W. Press et al, Numerical Recipes in C (2002)
    .. [3] Scargle, J.D. 1982, ApJ 263:835-853
    """
    # delegate to astropy's Lomb-Scargle
    ls = LombScargle(t, y, dy, fit_mean=generalized, center_data=subtract_mean)
    frequency = np.asarray(omega) / (2 * np.pi)
    p_omega = ls.power(frequency, method='cython')

    if significance is not None:
        N = t.size
        M = 2 * N
        z = (-2.0 / (N - 1.)
             * np.log(1 - (1 - np.asarray(significance)) ** (1. / M)))
        return p_omega, z
    else:
        return p_omega


@deprecated('0.4', alternative='astropy.stats.LombScargle.false_alarm_probability',
            warning_type=AstroMLDeprecationWarning)
def lomb_scargle_bootstrap(t, y, dy, omega,
                           generalized=True, subtract_mean=True,
                           N_bootstraps=100, random_state=None):
    """Use a bootstrap analysis to compute Lomb-Scargle significance

    Parameters
    ----------
    The first set of parameters are passed to the lomb_scargle algorithm

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

    Remaining parameters control the bootstrap

    N_bootstraps : int
        number of bootstraps
    random_state : None, int, or RandomState object
        random seed, or random number generator

    Returns
    -------
    D : ndarray
        distribution of the height of the highest peak
    """
    random_state = check_random_state(random_state)
    t = np.asarray(t)
    y = np.asarray(y)
    dy = np.asarray(dy) + np.zeros_like(y)

    D = np.zeros(N_bootstraps)

    for i in range(N_bootstraps):
        ind = random_state.randint(0, len(y), len(y))
        p = lomb_scargle(t, y[ind], dy[ind], omega,
                         generalized=generalized, subtract_mean=subtract_mean)
        D[i] = p.max()

    return D


def lomb_scargle_AIC(P, y, dy, n_harmonics=1):
    """Compute the AIC for a Lomb-Scargle Periodogram

    Parameters
    ----------
    P : array_like
        lomb-scargle power
    y : array_like
        observations
    dy : array_like
        errors
    n_harmonics : int (optional)
        the number of harmonics used in the Lomb-Scargle fit. Default is 1

    Returns
    -------
    AIC : ndarray
        AIC value corresponding to values in P
    """
    P, y, dy = map(np.asarray(P, y, dy))
    w = 1. / dy ** 2
    mu = np.dot(w, y) / w.sum()
    N = len(y)
    return np.sum(((y_obs - mu) / dy) ** 2) * P - (2 * n_harmonics + 1) * 2


def lomb_scargle_BIC(P, y, dy, n_harmonics=1):
    """Compute the BIC for a Lomb-Scargle Periodogram

    Parameters
    ----------
    P : array_like
        lomb-scargle power
    y : array_like
        observations
    dy : array_like
        errors
    n_harmonics : int (optional)
        the number of harmonics used in the Lomb-Scargle fit. Default is 1

    Returns
    -------
    BIC : ndarray
        BIC value corresponding to values in P
    """
    P, y, dy = map(np.asarray, (P, y, dy))
    w = 1. / dy ** 2
    mu = np.dot(w, y) / w.sum()
    N = len(y)
    return np.sum(((y - mu) / dy) ** 2) * P - (2 * n_harmonics + 1) * np.log(N)


@deprecated('0.4', alternative='astropy.stats.LombScargle',
            warning_type=AstroMLDeprecationWarning)
def multiterm_periodogram(t, y, dy, omega, n_terms=3):
    """Perform a multiterm periodogram at each omega

    This calculates the chi2 for the best-fit least-squares solution
    for each frequency omega.

    Parameters
    ----------
    t : array_like
        sequence of times
    y : array_like
        sequence of observations
    dy : array_like
        sequence of observational errors
    omega : float or array_like
        frequencies at which to evaluate p(omega)

    Returns
    -------
    power : ndarray
        P = 1. - chi2 / chi2_0
        where chi2_0 is the chi-square for a simple mean fit to the data
    """
    # TODO: deprecate this
    ls = LombScargle(t, y, dy, nterms=n_terms)
    frequency = np.asarray(omega) / (2 * np.pi)
    return ls.power(frequency)


def search_frequencies(t, y, dy,
                       LS_func=lomb_scargle,
                       LS_kwargs=None,
                       initial_guess=25,
                       limit_fractions=[0.04, 0.3, 0.9, 0.99],
                       n_eval=10000,
                       n_retry=5,
                       n_save=50):
    """Utility Routine to find the best frequencies

    To find the best frequency with a Lomb-Scargle periodogram requires
    searching a large range of frequencies at a very fine resolution.
    This is an iterative routine that searches progressively finer
    grids to narrow-in on the best result.

    Parameters
    ----------
    t: array_like
        observed times
    y: array_like
        observed fluxes or magnitudes
    dy: array_like
        observed errors on y

    Other Parameters
    ----------------
    LS_func : function
        Function used to perform Lomb-Scargle periodogram.  The call signature
        should be LS_func(t, y, dy, omega, **kwargs)
        (Default is astroML.periodogram.lomb_scargle)
    LS_kwargs : dict
        dictionary of keyword arguments to pass to LS_func in addition to
        (t, y, dy, omega)
    initial_guess : float
        the initial guess of the best period
    limit_fractions : array_like
        the list of fractions to use when zooming in on peak possibilities.
        On the i^th iteration, with f_i = limit_fractions[i], the range
        probed around each candidate will be
        (candidate * f_i, candidate / f_i).
    n_eval : integer or list
        The number of point to evaluate in the range on each iteration.
        If n_eval is a list, it should have the same length as limit_fractions.
    n_retry : integer or list
        Number of top points to search on each iteration. If n_retry is a list,
        it should have the same length as limit_fractions.
    n_save : integer or list
        Number of evaluations to save on each iteration.
        If n_save is a list, it should have the same length as limit_fractions.

    Returns
    -------
    omega_top, power_top: ndarrays
        The saved values of omega and power.  These will have size
        1 + n_save * (1 + n_retry * len(limit_fractions))
        as long as n_save > n_retry
    """
    if LS_kwargs is None:
        LS_kwargs = dict()

    omega_best = [initial_guess]
    power_best = LS_func(t, y, dy, omega_best, **LS_kwargs)

    for (Ne, Nr, Ns, frac) in np.broadcast(n_eval, n_retry,
                                           n_save, limit_fractions):
        # make sure we explore differing regions
        log_ob = np.log(omega_best)
        width = 0.1 * np.log(frac)
        log_ob = np.floor(-log_ob / width).astype(int)
        indices = np.arange(len(log_ob))

        for i in range(Nr):
            if len(indices) == 0:
                break
            omega_try = omega_best[indices[-1]]
            non_duplicates = (log_ob != log_ob[-1])
            log_ob = log_ob[non_duplicates]
            indices = indices[non_duplicates]

            omega = np.linspace(omega_try * frac, omega_try / frac, Ne)
            power = LS_func(t, y, dy, omega, **LS_kwargs)
            i = np.argsort(power)[-Ns:]
            power_best = np.concatenate([power_best, power[i]])
            omega_best = np.concatenate([omega_best, omega[i]])

        i = np.argsort(power_best)
        power_best = power_best[i]
        omega_best = omega_best[i]

    i = np.argsort(omega_best)
    return omega_best[i], power_best[i]


class MultiTermFit:
    """Multi-term Fourier fit to a light curve

    Parameters
    ----------
    omega : float
        angular frequency of the fundamental mode
    n_terms : int
        the number of Fourier modes to use in the fit
    """
    def __init__(self, omega, n_terms):
        self.omega = omega
        self.n_terms = n_terms

    def _make_X(self, t):
        t = np.asarray(t)
        k = np.arange(1, self.n_terms + 1)
        X = np.hstack([np.ones(t[:, None].shape),
                       np.sin(k * self.omega * t[:, None]),
                       np.cos(k * self.omega * t[:, None])])
        return X

    def fit(self, t, y, dy):
        """Fit multiple Fourier terms to the data

        Parameters
        ----------
        t: array_like
            observed times
        y: array_like
            observed fluxes or magnitudes
        dy: array_like
            observed errors on y

        Returns
        -------
        self :
            The MultiTermFit object is  returned
        """
        t = np.asarray(t)
        y = np.asarray(y)
        dy = np.asarray(dy)

        X_scaled = self._make_X(t) / dy[:, None]
        y_scaled = y / dy

        self.t_ = t
        self.w_ = np.linalg.solve(np.dot(X_scaled.T, X_scaled),
                                  np.dot(X_scaled.T, y_scaled))
        return self

    def predict(self, Nphase, return_phased_times=False, adjust_offset=True):
        """Compute the phased fit, and optionally return phased times

        Parameters
        ----------
        Nphase : int
            Number of terms to use in the phased fit
        return_phased_times : bool
            If True, then return a phased version of the input times
        adjust_offset : bool
            If true, then shift results so that the minimum value is at phase 0

        Returns
        -------
        phase, y_fit : ndarrays
            The phase and y value of the best-fit light curve
        phased_times : ndarray
            The phased version of the training times.  Returned if
            return_phased_times is set to  True.
        """
        phase_fit = np.linspace(0, 1, Nphase + 1)[:-1]

        X_fit = self._make_X(2 * np.pi * phase_fit / self.omega)
        y_fit = np.dot(X_fit, self.w_)
        i_offset = np.argmin(y_fit)

        if adjust_offset:
            y_fit = np.concatenate([y_fit[i_offset:], y_fit[:i_offset]])

        if return_phased_times:
            if adjust_offset:
                offset = phase_fit[i_offset]
            else:
                offset = 0
            phased_times = (self.t_ * self.omega * 0.5 / np.pi - offset) % 1

            return phase_fit, y_fit, phased_times

        else:
            return phase_fit, y_fit
