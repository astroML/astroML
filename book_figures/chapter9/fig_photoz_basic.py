"""
Photometric Redshifts via Linear Regression
-------------------------------------------

Linear Regression for photometric redshifts
We could use sklearn.linear_model.LinearRegression, but to be more
transparent, we'll do it by hand using linear algebra.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import itertools

import numpy as np
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import euclidean_distances

from astroML.datasets import fetch_sdss_specgals

np.random.seed(0)

data = fetch_sdss_specgals()

# put magnitudes in a matrix
# with a constant (for the intercept) at position zero
mag = np.vstack([np.ones(data.shape)]
                + [data['modelMag_%s' % f] for f in 'ugriz']).T
z = data['z']

# train on ~60,000 points
mag_train = mag[::10]
z_train = z[::10]

# test on ~6,000 distinct points
mag_test = mag[1::100]
z_test = z[1::100]


def plot_results(z, z_fit, plotlabel=None,
                 xlabel=True, ylabel=True):
    plt.scatter(z, z_fit, s=1, lw=0, c='k')
    plt.plot([-0.1, 0.4], [-0.1, 0.4], ':k')
    plt.xlim(-0.05, 0.4001)
    plt.ylim(-0.05, 0.4001)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))

    if plotlabel:
        plt.text(0.03, 0.97, plotlabel,
                 ha='left', va='top', transform=ax.transAxes)

    if xlabel:
        plt.xlabel(r'$\rm z_{true}$', fontsize=16)
    else:
        plt.gca().xaxis.set_major_formatter(plt.NullFormatter())

    if ylabel:
        plt.ylabel(r'$\rm z_{fit}$', fontsize=16)
    else:
        plt.gca().yaxis.set_major_formatter(plt.NullFormatter())


def combinations_with_replacement(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    for indices in itertools.product(range(n), repeat=r):
        if sorted(indices) == list(indices):
            yield tuple(pool[i] for i in indices)


def poly_features(X, p):
    """Compute polynomial features

    Parameters
    ----------
    X: array_like
        shape (n_samples, n_features)
    p: int
        degree of polynomial
    Returns
    -------
    X_p: array
        polynomial feature matrix
    """
    X = np.asarray(X)
    N, D = X.shape
    ind = list(combinations_with_replacement(range(D), p))
    X_poly = np.empty((X.shape[0], len(ind)))

    for i in range(len(ind)):
        X_poly[:, i] = X[:, ind[i]].prod(1)

    return X_poly


def gaussian_RBF_features(X, centers, widths):
    """Compute gaussian Radial Basis Function features

    Parameters
    ----------
    X: array_like
        shape (n_samples, n_features)
    centers: array_like
        shape (n_centers, n_features)
    widths: array_like
        shape (n_centers, n_features) or (n_centers,)
    Returns
    -------
    X_RBF: array
        RBF feature matrix, shape=(n_samples, n_centers)
    """
    X, centers, widths = map(np.asarray, (X, centers, widths))
    if widths.ndim == 1:
        widths = widths[:, np.newaxis]
    return np.exp(-0.5 * ((X[:, np.newaxis, :]
                           - centers) / widths) ** 2).sum(-1)

plt.figure(figsize=(8, 8))
plt.subplots_adjust(hspace=0.05, wspace=0.05,
                    left=0.1, right=0.95,
                    bottom=0.1, top=0.95)

#----------------------------------------------------------------------
# first do a simple linear regression between the r-band and redshift,
# ignoring uncertainties
ax = plt.subplot(221)
X_train = mag_train[:, [0, 3]]
X_test = mag_test[:, [0, 3]]
z_fit = LinearRegression().fit(X_train, z_train).predict(X_test)
plot_results(z_test, z_fit,
             plotlabel='Linear Regression:\n r-band',
             xlabel=False)

#----------------------------------------------------------------------
# next do a linear regression with all bands
ax = plt.subplot(222)
z_fit = LinearRegression().fit(mag_train, z_train).predict(mag_test)
plot_results(z_test, z_fit, plotlabel="Linear Regression:\n ugriz bands",
             xlabel=False, ylabel=False)

#----------------------------------------------------------------------
# next do a 3rd-order polynomial regression with all bands
ax = plt.subplot(223)
X_train = poly_features(mag_train, 3)
X_test = poly_features(mag_test, 3)
z_fit = LinearRegression().fit(X_train, z_train).predict(X_test)
plot_results(z_test, z_fit, plotlabel="3rd order Polynomial\nRegression")

#----------------------------------------------------------------------
# next do a radial basis function regression with all bands
ax = plt.subplot(224)

# remove bias term
mag = mag[:, 1:]
mag_train = mag_train[:, 1:]
mag_test = mag_test[:, 1:]

centers = mag[np.random.randint(mag.shape[0], size=100)]
centers_dist = euclidean_distances(centers, centers, squared=True)
widths = np.sqrt(centers_dist[:, :10].mean(1))

X_train = gaussian_RBF_features(mag_train, centers, widths)
X_test = gaussian_RBF_features(mag_test, centers, widths)
z_fit = LinearRegression().fit(X_train, z_train).predict(X_test)
plot_results(z_test, z_fit, plotlabel="Gaussian Basis Function\nRegression",
             ylabel=False)

plt.show()
