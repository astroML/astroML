"""
Gaussian Naive Bayes Classification
-----------------------------------

Plot a histogram of the SDSS training & test colors, with gaussian fits
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning for Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
import pylab as pl

from sklearn import naive_bayes

from astroML.pdf import GaussianProbability
from astroML.datasets import fetch_sdss_colors_train, fetch_sdss_colors_test
from astroML.plotting import multidensity


# define a function to print the results
def output_results(y_test, y_pred):
    labels = ['QSO', 'star']
    for i, label in enumerate(labels):
        mask = (y_test == i)
        num_total = mask.sum()
        num_correct = (y_test[mask] == y_pred[mask]).sum()
        pct = num_correct * 100. / num_total
        print ("  %(label)ss : %(num_correct)i/%(num_total)i "
               "match (%(pct).2f%%)" % locals())

# load the data
X_train, y_train = fetch_sdss_colors_train()
X_test, y_test = fetch_sdss_colors_test()

# Determine the prior
num_qsos = np.sum(y_train == 0)
num_stars = np.sum(y_train == 1)
num_total = len(y_train)
prior = np.array([num_qsos, num_stars]) * 1. / num_total

# Compute the log-likelihoods of the test points
logL = np.zeros((2, len(y_test)))
for i in range(2):
    ind = np.where(y_train == i)
    for j in range(4):
        P = GaussianProbability(X_train[ind[0], j])
        logL[i] += np.log(P(X_test[:, j]))
y_pred = np.argmax(logL + np.log(prior)[:, None], 0)

print "Results for by-hand method:"
output_results(y_test, y_pred)

# Alternatively, we can do this with scikit-learn in a couple lines
gnb = naive_bayes.GaussianNB().fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print
print "Results for scikit-learn method:"
output_results(y_test, y_pred)


# plot a dummy plot for testing purposes
def plot_correlation(X, y):
    flag = (y == 1)
    stars = X[flag]
    qsos = X[~flag]
    labels = ['u-g', 'g-r', 'r-i', 'i-z']

    bins = [np.linspace(-0.5, 3.5, 100),
            np.linspace(0, 2, 100),
            np.linspace(-0.2, 1.8, 100),
            np.linspace(-0.2, 1.0, 100)]
    multidensity(stars, labels, bins=bins)
    pl.suptitle('Stars', fontsize=18)
    pl.gcf().caption = ("Density Plot for Stellar Colors\n"
                        "-------------------------------\n\n"
                        "This is the density plot of stellar colors.\n")

    bins = [np.linspace(-0.5, 1, 100),
            np.linspace(-0.5, 1, 100),
            np.linspace(-0.5, 1, 100),
            np.linspace(-0.5, 1, 100)]
    multidensity(qsos, labels, bins=bins)
    pl.suptitle('QSOs', fontsize=18)

plot_correlation(X_train, y_train)
pl.show()
