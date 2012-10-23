import numpy as np
from astroML.utils import combinations_with_replacement


class LinearRegression:
    """Simple Linear Regression with errors in y

    This is a stripped-down version of sklearn.linear_model.LinearRegression
    which can correctly accounts for errors in the y variable

    Parameters
    ----------
    fit_intercept : bool (optional)
        if True (default) then fit the intercept of the data

    Notes
    -----
    This implementation may be compared to that in
    sklearn.linear_model.LinearRegression.
    The difference is that here errors are
    """
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def _process_Xy(self, X, y, dy):
        X = self._process_X(X)
        y = np.asarray(y, dtype=float)
        dy = np.atleast_1d(dy)

        return X / dy[:, None], y / dy

    def _process_X(self, X):
        X = np.asarray(X, dtype=float)
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X

    def fit(self, X, y, dy=1):
        self.y_ = np.asarray(y)
        self.X_ = np.asarray(X)
        self.dy_ = dy

        X_fit, y_fit = self._process_Xy(self.X_, self.y_, dy)

        self.coef_ = np.linalg.solve(np.dot(X_fit.T, X_fit),
                                     np.dot(X_fit.T, y_fit))
        return self

    def predict(self, X):
        return np.dot(self._process_X(X), self.coef_)


class PolynomialRegression(LinearRegression):
    """Polynomial Regression with errors in y

    Parameters
    ----------
    degree : int
        degree of the polynomial.
    fit_intercept : bool (optional)
        if True (default) then fit the intercept of the data
    """
    def __init__(self, degree=1, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.degree = degree

    def _process_X(self, X):
        X = np.asarray(X, dtype=float)
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        ind = np.array(list(combinations_with_replacement(range(X.shape[1]),
                                                          self.degree)))

        X = X[:, ind].prod(-1)

        if self.fit_intercept:
            return X
        else:
            return X[:, 1:]


class BasisFunctionRegression(LinearRegression):
    """Basis Function with errors in y

    Parameters
    ----------
    fit_intercept : bool (optional)
        if True (default) then fit the intercept of the data
    basis_func : str or function
        specify the basis function to use.  This should take an input matrix
        of size (n_samples, n_features), along with optional parameters,
        and return a matrix of size (n_samples, n_bases).
    **kwargs :
        extra arguments and keyword arguments are passed to the basis
        function
    """
    def __init__(self, basis_func='gaussian', fit_intercept=True, **kwargs):
        if basis_func == 'gaussian':
            self.basis_func = gaussian_basis
            if ('mu' not in kwargs) or ('sigma' not in kwargs):
                raise ValueError('For gaussian basis, mu and sigma must '
                                 'be specified')
        elif callable(basis_func):
            self.basis_func = basis_func
        else:
            raise ValueError("basis_func not understood")

        self.fit_intercept = fit_intercept
        self.kwargs = kwargs

    def _process_X(self, X):
        X = self.basis_func(X, **self.kwargs)

        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        return X


#------------------------------------------------------------
# Basis functions
def gaussian_basis(X, mu, sigma):
    """Gaussian Basis function

    Parameters
    ----------
    X : array_like
        input data: shape = (n_samples, n_features)
    mu : array_like
        means of bases, shape = (n_bases, n_features)
    sigma : float or array_like
        must broadcast to shape of mu

    Returns
    -------
    Xg : ndarray
        shape = (n_samples, n_bases)
    """
    X = np.asarray(X)
    mu = np.atleast_2d(mu)
    sigma = np.atleast_2d(sigma)

    n_samples, n_features = X.shape
    n_bases = mu.shape[0]

    if mu.shape[1] != n_features:
        raise ValueError('shape of mu must match shape of X')

    r = (((X[:, None, :] - mu) / sigma) ** 2).sum(2)
    Xg = np.exp(-0.5 * r)
    Xg *= 1. / np.sqrt(2 * np.pi) / sigma.prod(1)

    return Xg
