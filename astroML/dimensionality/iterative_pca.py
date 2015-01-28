import sys

import numpy as np

from scipy.linalg import solve


def iterative_pca(X, M, n_ev=5, n_iter=15, norm=None, full_output=False):
    """
    Parameters
    ----------
    X: ndarray, shape = (n_samples, n_features)
        input data
    M: ndarray, bool, shape = (n_samples, n_features)
        mask for input data.  where mask == True, the spectrum is unconstrained
    n_ev: int
        number of eigenvectors to use in reconstructing masked regions
    n_iter: int
        number of iterations to find eigenvectors
    norm: string
        what type of normalization to use on the data. Options are
        - None : no normalization
        - 'L1' : L1-norm
        - 'L2' : L2-norm
    full_output: boolean (optional)
        if False (default) return only the reconstructed data X_recons
        if True, return the full information (see below)

    Returns
    -------
    X_recons: ndarray, shape = (n_samples, n_features)
        data with masked regions reconstructed

    mu: ndarray, shape = (n_features,)
        mean of data
    evecs: ndarray, shape = (min(n_samples, n_features), n_features)
        eigenvectors of the reconstructed data
    evals: ndarray, size = min(n_samples, n_features)
        eigenvalues of the reconstructed data
    norms: ndarray, size = n_samples
        normalization of each input
    coeffs: ndarray, size = (n_samples, n_ev)
        coefficients used to reconstruct X
    """
    X = np.asarray(X, dtype=np.float)
    M = np.asarray(M, dtype=np.bool)

    if X.shape != M.shape:
        raise ValueError('X and M must have the same shape')

    n_samples, n_features = X.shape

    if np.any(M.sum(0) == n_samples):
        raise ValueError('Some features are masked in all samples')

    if type(norm) == str:
        norm = norm.upper()

    if norm not in (None, 'none', 'L1', 'L2'):
        raise ValueError('unrecognized norm: %s' % norm)

    notM = (~M)

    X_recons = X.copy()
    X_recons[M] = 0

    # as an initial guess, we'll fill-in masked regions with the mean
    # of the rest of the sample
    if norm is None:
        mu = (X_recons * notM).sum(0) / notM.sum(0)
        mu = mu * np.ones([n_samples, 1])
        X_recons[M] = mu[M]
    else:
        # since we're normalizing each spectrum, and the norm depends on
        # the filled-in values, we need to iterate a few times to make
        # sure things are consistent.
        for i in range(n_iter):
            # normalize
            if norm == 'L1':
                X_recons /= np.sum(X_recons, 1)[:, None]
            else:
                X_recons /= np.sqrt(np.sum(X_recons ** 2, 1))[:, None]

            # find the mean
            mu = (X_recons * notM).sum(0) / notM.sum(0)
            mu = mu * np.ones([n_samples, 1])
            X_recons[M] = mu[M]

    # Matrix of coefficients
    coeffs = np.zeros((n_samples, n_ev))

    # Now we iterate through, using the principal components to reconstruct
    #  these regions.
    for i in range(n_iter):
        sys.stdout.write(' PCA iteration %i / %i\r' % (i + 1, n_iter))
        sys.stdout.flush()

        # normalize the data
        if norm == 'L1':
            X_recons /= np.sum(X_recons, 1)[:, None]
        else:
            X_recons /= np.sqrt(np.sum(X_recons ** 2, 1))[:, None]

        # now compute the principal components
        mu = X_recons.mean(0)
        X_centered = X_recons - mu
        U, S, VT = np.linalg.svd(X_centered, full_matrices=False)

        # perform a least-squares fit to estimate the coefficients of the
        # first n_ev eigenvectors for each data point.
        # The eigenvectors are in the rows of the matrix VT.
        # The coefficients are given by
        #  a_n = [V_n^T W V_n]^(-1) V_n W x
        # Such that x can be reconstructed via
        #  x_n = V_n a_n
        # Variables here are:
        #  x   : vector length n_features. This is a data point to be
        #        reconstructed
        #  a_n : vector of length n.  These are the reconstruction weights
        #  V_n : eigenvector matrix of size (n_features, n).
        #  W   : diagonal weight matrix of size (n_features, n_features)
        #        such that W[i,i] = M[i]
        #  x_n : vector of length n_features which approximates x
        VWx = np.dot(VT[:n_ev], (notM * X_centered).T)
        for i in range(n_samples):
            VWV = np.dot(VT[:n_ev], (notM[i] * VT[:n_ev]).T)
            coeffs[i] = solve(VWV, VWx[:, i], sym_pos=True, overwrite_a=True)

        X_fill = mu + np.dot(coeffs, VT[:n_ev])
        X_recons[M] = X_fill[M]
    sys.stdout.write('\n')

    # un-normalize X_recons
    norms = np.zeros(n_samples)
    for i in range(n_samples):
        ratio_i = X[i][notM[i]] / X_recons[i][notM[i]]
        norms[i] = ratio_i[~np.isnan(ratio_i)][0]
        X_recons[i] *= norms[i]

    if full_output:
        return X_recons, mu, VT, S, norms, coeffs
    else:
        return X_recons
