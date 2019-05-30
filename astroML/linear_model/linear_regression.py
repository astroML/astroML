import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge


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


class LinearRegression:
    """Simple Linear Regression with errors in y

    This is a stripped-down version of sklearn.linear_model.LinearRegression
    which can correctly accounts for errors in the y variable

    Parameters
    ----------
    fit_intercept : bool (optional)
        if True (default) then fit the intercept of the data
    regularization : string (optional)
        ['l1'|'l2'|'none'] Use L1 (Lasso) or L2 (Ridge) regression
    kwds: dict
        additional keyword arguments passed to sklearn estimators:
        LinearRegression, Lasso (L1), or Ridge (L2)

    Notes
    -----
    This implementation may be compared to that in
    sklearn.linear_model.LinearRegression.
    The difference is that here errors are
    """
    _regressors = {'none' : LinearRegression,
                   'l1' : Lasso,
                   'l2' : Ridge}

    def __init__(self, fit_intercept=True, regularization='none', kwds=None):
        if regularization.lower() not in ['l1', 'l2', 'none']:
            raise ValueError("regularization='{}' not recognized"
                             "".format(regularization))
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self.kwds = kwds

    def _transform_X(self, X):
        X = np.asarray(X)
        if self.fit_intercept:
            X = np.hstack([np.ones([X.shape[0], 1]), X])
        return X

    @staticmethod
    def _scale_by_error(X, y, y_error=1):
        """Scale regression by error on y"""
        X = np.atleast_2d(X)
        y = np.asarray(y)
        y_error = np.asarray(y_error)

        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]

        if y_error.ndim == 0:
            return X / y_error, y / y_error

        elif y_error.ndim == 1:
            assert y_error.shape == y.shape
            X_out, y_out = X / y_error[:, None], y / y_error

        elif y_error.ndim == 2:
            assert y_error.shape == (y.size, y.size)
            evals, evecs = np.linalg.eigh(y_error)
            X_out = np.dot(evecs * (evals ** -0.5),
                           np.dot(evecs.T, X))
            y_out = np.dot(evecs * (evals ** -0.5),
                           np.dot(evecs.T, y))
        else:
            raise ValueError("shape of y_error does not match that of y")

        return X_out, y_out

    def _choose_regressor(self):
        model = self._regressors.get(self.regularization.lower(), None)
        if model is None:
            raise ValueError("regularization='{}' unrecognized"
                             "".format(self.regularization))
        return model

    def fit(self, X, y, y_error=1):
        kwds = {}
        if self.kwds is not None:
            kwds.update(self.kwds)
        kwds['fit_intercept'] = False

        model = self._choose_regressor()
        self.clf_ = model(**kwds)

        X = self._transform_X(X)
        X, y = self._scale_by_error(X, y, y_error)

        self.clf_.fit(X, y)
        return self

    def predict(self, X):
        X = self._transform_X(X)
        return self.clf_.predict(X)

    @property
    def coef_(self):
        return self.clf_.coef_


class PolynomialRegression(LinearRegression):
    """Polynomial Regression with errors in y

    Parameters
    ----------
    degree : int
        degree of the polynomial.
    interaction_only : bool (optional)
        If true, only interaction features are produced: features that are
        products of at most ``degree`` *distinct* input features (so not
        ``x[1] ** 2``, ``x[0] * x[2] ** 3``, etc.).
    fit_intercept : bool (optional)
        if True (default) then fit the intercept of the data
    regularization : string (optional)
        ['l1'|'l2'|'none'] Use L1 (Lasso) or L2 (Ridge) regression
    kwds: dict
        additional keyword arguments passed to sklearn estimators:
        LinearRegression, Lasso (L1), or Ridge (L2)
    """
    def __init__(self, degree=1, interaction_only=False,
                 fit_intercept=True,
                 regularization='none', kwds=None):
        self.degree = degree
        self.interaction_only = interaction_only
        LinearRegression.__init__(self, fit_intercept, regularization, kwds)

    def _transform_X(self, X):
        trans = PolynomialFeatures(degree=self.degree,
                                   interaction_only=self.interaction_only,
                                   include_bias=self.fit_intercept)
        return trans.fit_transform(X)


class BasisFunctionRegression(LinearRegression):
    """Basis Function with errors in y

    Parameters
    ----------
    basis_func : str or function
        specify the basis function to use.  This should take an input matrix
        of size (n_samples, n_features), along with optional parameters,
        and return a matrix of size (n_samples, n_bases).
    fit_intercept : bool (optional)
        if True (default) then fit the intercept of the data
    regularization : string (optional)
        ['l1'|'l2'|'none'] Use L1 (Lasso) or L2 (Ridge) regression
    kwds: dict
        additional keyword arguments passed to sklearn estimators:
        LinearRegression, Lasso (L1), or Ridge (L2)
    """
    _basis_funcs = {'gaussian': gaussian_basis}

    def __init__(self, basis_func='gaussian', fit_intercept=True,
                 regularization='none', kwds=None, **kwargs):
        self.basis_func = basis_func
        self.kwargs = kwargs
        LinearRegression.__init__(self, fit_intercept, regularization, kwds)

    def _transform_X(self, X):
        if callable(self.basis_func):
            basis_func = self.basis_func
        else:
            basis_func = self._basis_funcs.get(self.basis_func, None)

        X = basis_func(X, **self.kwargs)
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        return X
