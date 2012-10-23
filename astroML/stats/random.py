"""
Statistics for astronomy
"""
import numpy as np
from scipy.stats.distributions import rv_continuous


def bivariate_normal(mu=[0, 0], sigma_1=1, sigma_2=1, alpha=0,
                     size=None, return_cov=False):
    """Sample points from a 2D normal distribution

    Parameters
    ----------
    mu : array-like (length 2)
        The mean of the distribution
    sigma_1 : float
        The unrotated x-axis width
    sigma_2 : float
        The unrotated y-axis width
    alpha : float
        The rotation counter-clockwise about the origin
    size : tuple of ints, optional
        Given a shape of, for example, ``(m,n,k)``, ``m*n*k`` samples are
        generated, and packed in an `m`-by-`n`-by-`k` arrangement.  Because
        each sample is `N`-dimensional, the output shape is ``(m,n,k,N)``.
        If no shape is specified, a single (`N`-D) sample is returned.
    return_cov : boolean, optional
        If True, return the computed covariance matrix.

    Returns
    -------
    out : ndarray
        The drawn samples, of shape *size*, if that was provided.  If not,
        the shape is ``(N,)``.

        In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
        value drawn from the distribution.

    cov : ndarray
        The 2x2 covariance matrix.  Returned only if return_cov == True.

    Notes
    -----
    This function works by computing a covariance matrix from the inputs,
    and calling ``np.random.multivariate_normal()``.  If the covariance
    matrix is available, this function can be called directly.
    """
    # compute covariance matrix
    sigma_xx = ((sigma_1 * np.cos(alpha)) ** 2
                + (sigma_2 * np.sin(alpha)) ** 2)
    sigma_yy = ((sigma_1 * np.sin(alpha)) ** 2
                + (sigma_2 * np.cos(alpha)) ** 2)
    sigma_xy = (sigma_1 ** 2 - sigma_2 ** 2) * np.sin(alpha) * np.cos(alpha)

    cov = np.array([[sigma_xx, sigma_xy],
                    [sigma_xy, sigma_yy]])

    # draw points from the distribution
    x = np.random.multivariate_normal(mu, cov, size)

    if return_cov:
        return x, cov
    else:
        return x


#----------------------------------------------------------------------
# Define some new distributions based on rv_continuous
class trunc_exp_gen(rv_continuous):
    """A truncated positive exponential continuous random variable.

    The probability distribution is::

       p(x) ~ exp(k * x)   between a and b
            = 0            otherwise

    The arguments are (a, b, k)

    %(before_notes)s

    %(example)s

    """
    def _argcheck(self, a, b, k):
        self._const = k / (np.exp(k * b) - np.exp(k * a))
        return (a != b) and not np.isinf(k)

    def _pdf(self, x, a, b, k):
        pdf = self._const * np.exp(k * x)
        pdf[(x < a) | (x > b)] = 0
        return pdf

    def _rvs(self, a, b, k):
        y = np.random.random(self._size)
        return (1. / k) * np.log(1 + y * k / self._const)

trunc_exp = trunc_exp_gen(name="trunc_exp", shapes='a, b, k')


class linear_gen(rv_continuous):
    """A truncated positive exponential continuous random variable.

    The probability distribution is::

       p(x) ~ c * x + d   between a and b
            = 0             otherwise

    The arguments are (a, b, c).  d is set by the normalization

    %(before_notes)s

    %(example)s

    """
    def _argcheck(self, a, b, c):
        return (a != b) and not np.isinf(c)

    def _pdf(self, x, a, b, c):
        d = 1. / (b - a) - 0.5 * c * (b + a)
        pdf = c * x + d
        pdf[(x < a) | (x > b)] = 0
        return pdf

    def _rvs(self, a, b, c):
        mu = 0.5 * (a + b)
        W = (b - a)

        x0 = 1. / c / W - mu
        r = np.random.random(self._size)
        return -x0 + np.sqrt(2. * r / c + a * a
                             + 2. * a * x0 + x0 * x0)

linear = linear_gen(name="linear", shapes='a, b, c')
