import numpy as np
from .linear_regression import gaussian_basis
from sklearn.metrics import pairwise_kernels


class NadarayaWatson:
    """Nadaraya-Watson Kernel Regression

    This is basically a gaussian-weighted moving average of points

    Parameters
    ----------
    kernel : string
        kernel is either "gaussian", or one of the kernels available in
        sklearn.metrics.pairwise.
    h : float or array_like
        width of kernel.  If array, its length must be the number of
        dimensions in the training data

    Additional keyword arguments are passed to the kernel.
    """
    def __init__(self, kernel='gaussian', h=None, **kwargs):
        self.kernel = kernel
        self.h = h
        self.kwargs = kwargs

    def fit(self, X, y, dy=1):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.dy = np.atleast_1d(dy)
        return self

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError('X must be two-dimensional')

        if X.shape[1] != self.X.shape[1]:
            raise ValueError('dimensions of X do not match training dimension')

        if self.kernel == 'gaussian':
            # wrangle gaussian into scikit-learn's 'rbf' kernel
            h = np.asarray(self.h)
            gamma = 0.5 / h / h
            K = pairwise_kernels(X, self.X, metric='rbf', gamma=gamma)

        else:
            K = pairwise_kernels(X, self.X, metric=self.kernel, **self.kwargs)

        K /= self.dy ** 2

        return (K * self.y).sum(1) / K.sum(1)
