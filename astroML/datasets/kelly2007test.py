"""
Generate input data based on Section 7 in Kelly 2007
"""

import numpy as np
from scipy.stats.distributions import rv_continuous


__all__ = ['simulation_kelly']


class simulation_dist(rv_continuous):

    def _pdf(self, x):
        #  eq 110
        return 0.796248 * np.exp(x) / (1 + np.exp(2.75 * x))


def simulation_kelly(size=50, low=-10, high=10, alpha=1, beta=0.5,
                     epsilon=(0, 0.75), scalex=1, scaley=1, multidim=1,
                     ksi=None, eta=None):
    """
    Data simulator from Kelly 2007

    Parameters
    ==========
    size : int
        Number of datapoints to be generated
    alpha : float
        Regression coefficient defined in eq 1
    beta : float
        Regression coefficient defined in eq 1
    epsilon : tupple of floats
        Mean and standard deviation of normally distributed intrinsic scatter
    scalex : float
        Scale parameter for x measurement errors
    scaley : float
        Scale parameter for y measurement errors
    multidim : int
        Dimension of multivariate data
    ksi : array
        If both ``ksi`` and ``eta`` are not None, use them to generate the measured
        ``xi`` and ``yi`` values and their measurement errors using the scale parameters.
    eta : array
        If both ``ksi`` and ``eta`` are not None, use them to generate the measured
        ``xi`` and ``yi`` values and their measurement errors using the scale parameters.
    low : float
        Lower bound of the support of the distribution (see eq. 110 in Kelly 2007).
    high : float
        Upper bound of the support of the distribution (see eq. 110 in Kelly 2007).

    Returns
    =======
    ksi
    eta
    xi
    yi
    xi_error
    yi_error
    alpha_in
    beta_in

    Notes
    =====

    """
    eps = np.random.normal(epsilon[0], scale=epsilon[1], size=size)
    beta = np.atleast_1d(beta)

    if ksi is None and eta is None:
        dist = simulation_dist(a=low, b=high)
        ksi = dist.rvs(size=(multidim, size))

        # eq 1
        eta = alpha + np.dot(beta, ksi) + eps

    tau = np.var(ksi)

    t = scalex * tau
    s = scaley * epsilon[1]

    # measurement errors from scaled inverse chi2 with df=5
    sigma_x = 5 * t / np.random.chisquare(df=5, size=(multidim, size))
    sigma_y = 5 * s / np.random.chisquare(df=5, size=size)

    x = np.random.normal(ksi, sigma_x)
    y = np.random.normal(eta, sigma_y)

    return ksi, eta, x, y, sigma_x, sigma_y, alpha, beta
