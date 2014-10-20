import numpy as np
from scipy import optimize


def TLS_logL(v, X, dX):
    """Compute the total least squares log-likelihood

    This uses Hogg et al eq. 29-32

    Parameters
    ----------
    v : ndarray
        The normal vector to the linear best fit.  shape=(D,).
        Note that the magnitude |v| is a stand-in for the intercept.
    X : ndarray
        The input data.  shape = [N, D]
    dX : ndarray
        The covariance of the errors for each point.
        For diagonal errors, the shape = (N, D) and the entries are
        dX[i] = [sigma_x1, sigma_x2 ... sigma_xD]
        For full covariance, the shape = (N, D, D) and the entries are
        dX[i] = Cov(X[i], X[i]), the full error covariance.

    Returns
    -------
    logL : float
        The log-likelihood of the model v given the data.

    Notes
    -----
    This implementation follows Hogg 2010, arXiv 1008.4686
    """
    # check inputs
    X, dX, v = map(np.asarray, (X, dX, v))
    N, D = X.shape
    assert v.shape == (D,)
    assert dX.shape in ((N, D), (N, D, D))

    v_norm = np.linalg.norm(v)
    v_hat = v / v_norm

    # eq. 30
    Delta = np.dot(X, v_hat) - v_norm

    # eq. 31
    if dX.ndim == 2:
        # diagonal covariance
        Sig2 = np.sum(dX * v_hat ** 2, 1)
    else:
        # full covariance
        Sig2 = np.dot(np.dot(v_hat, dX), v_hat)

    return (-0.5 * np.sum(np.log(2 * np.pi * Sig2))
            - np.sum(0.5 * Delta ** 2 / Sig2))
