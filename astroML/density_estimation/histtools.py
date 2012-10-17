"""
Tools for working with distributions
"""
import numpy as np
from scipy.special import gammaln
from scipy import optimize

def scotts_bin_width(data):
    r"""Return the optimal histogram bin width using Scott's rule:

    Parameters
    ----------
    data : array-like, ndim=1
        observed (one-dimensional) data

    Returns
    -------
    width : float
        optimal bin width using Scott's rule

    Notes
    -----
    The optimal bin width is

    .. math::
        \Delta_b = \frac{3.5\sigma}{n^{1/3}}

    where :math:`\sigma` is the standard deviation of the data, and
    :math:`n` is the number of data points.

    See Also
    --------
    knuth_nbins
    freedman_bin_width
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("data should be one-dimensional")

    n = data.size
    sigma = np.std(data)

    return 3.5 * sigma * 1. / (n ** (1./ 3))

def freedman_bin_width(data):
    r"""Return the optimal histogram bin width using the Freedman-Diaconis rule 

    Parameters
    ----------
    data : array-like, ndim=1
        observed (one-dimensional) data

    Returns
    -------
    width : float
        optimal bin width using Scott's rule

    Notes
    -----
    The optimal bin width is

    .. math::
        \Delta_b = \frac{2(q_{75} - q_{25})}{n^{1/3}}

    where :math:`q_{N}` is the :math:`N` percent quartile of the data, and
    :math:`n` is the number of data points.

    See Also
    --------
    knuth_nbins
    scotts_bin_width
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("data should be one-dimensional")

    n = data.size
    if n < 4:
        raise ValueError("data should have more than three entries")

    indices = np.argsort(data)
    i25 = indices[n/4 - 1]
    i75 = indices[(3 * n) / 4 - 1]
    
    return 2 * (data[i75] - data[i25]) * 1. / (n ** (1./ 3))


class KnuthF:
    r"""Class which implements the function minimized by knuth_nbins

    Parameters
    ----------
    data : array-like, one dimension
        data to be histogrammed

    Notes
    -----
    the function F is given by

    .. math::
        F(M|x,I) = n\log(M) + \log\Gamma(\frac{M}{2})
        - M\log\Gamma(\frac{1}{2})
        - \log\Gamma(\frac{2n+M}{2})
        + \sum_{k=1}^M \log\Gamma(n_k + \frac{1}{2})

    where :math:`\Gamma` is the Gamma function, :math:`n` is the number of
    data points, :math:`n_k` is the number of measurements in bin :math:`k`.

    See Also
    --------
    knuth_nbins
    """
    def __init__(self, data):
        self.data = np.array(data, copy=True)
        if self.data.ndim != 1:
            raise ValueError("data should be 1-dimensional")
        self.data.sort()
        self.n = self.data.size

    def __call__(self, M):
        return self.eval(M)

    def eval(self, M):
        """Evaluate the Knuth function

        Parameters
        ----------
        M : integer
            number of bins

        Returns
        -------
        F : float
            evaluation of the Knuth function
        """
        M = int(M)
        if M <= 0:
            return -np.inf

        nk, edges = np.histogram(self.data, int(M))
    
        return (self.n * np.log(M)
                + gammaln(0.5 * M)
                - M * gammaln(0.5)
                - gammaln(self.n + 0.5 * M)
                + np.sum(gammaln(nk + 0.5)))
    

def knuth_nbins(data, return_width=False):
    r"""Return the optimal histogram bin width using Knuth's rule [1]_

    Parameters
    ----------
    data : array-like, ndim=1
        observed (one-dimensional) data

    Returns
    -------
    N : integer
        optimal number of bins
    width : float
        optimal bin width.  Only returned if return_width == True

    Notes
    -----
    The optimal bin width is the value M which maximizes the function

    .. math::
        F(M|x,I) = n\log(M) + \log\Gamma(\frac{M}{2})
        - M\log\Gamma(\frac{1}{2})
        - \log\Gamma(\frac{2n+M}{2})
        + \sum_{k=1}^M \log\Gamma(n_k + \frac{1}{2})

    where :math:`\Gamma` is the Gamma function, :math:`n` is the number of
    data points, :math:`n_k` is the number of measurements in bin :math:`k`.

    References
    ----------
    .. [1] Knuth, K.H. "Optimal Data-Based Binning for Histograms".
           arXiv:0605197, 2006

    See Also
    --------
    KnuthF
    freedman_bin_width
    scotts_bin_width
    """
    knuthF = KnuthF(data)

    dmin = knuthF.data[0]
    dmax = knuthF.data[-1]

    F = lambda M: -knuthF(M)

    width = freedman_bin_width(data)
    M0 = int((dmax - dmin) / width)

    M = int(optimize.fmin(F, M0)[0])

    if return_width:
        return M, (dmax - dmin) * 1. / M
    else:
        return M
