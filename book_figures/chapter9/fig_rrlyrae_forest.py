"""
Decision Tree Classification of photometry
------------------------------------------
Decision Tree photometric classification of rr-lyrae stars.  This uses averaged
photometry from the rr-lyrae catalog and stripe 82 standards catalogs.
"""
# Author: Jake VanderPlas
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general

from __future__ import print_function, division

import numpy as np
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from astroML.datasets import fetch_rrlyrae_mags, fetch_sdss_S82standards

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=True)


def load_rrlyrae_data():
    """Load the RR Lyrae data.
    This will be used in several examples.
    """
    #----------------------------------------------------------------------
    # Load data

    rrlyrae = fetch_rrlyrae_mags()
    standards = fetch_sdss_S82standards()

    # perform color cuts on standard stars
    # these come from eqns 1-4 of Sesar et al 2010, ApJ 708:717

    u_g = standards['mmu_u'] - standards['mmu_g']
    g_r = standards['mmu_g'] - standards['mmu_r']
    r_i = standards['mmu_r'] - standards['mmu_i']
    i_z = standards['mmu_i'] - standards['mmu_z']

    standards = standards[(u_g > 0.7) & (u_g < 1.35) &
                          (g_r > -0.15) & (g_r < 0.4) &
                          (r_i > -0.15) & (r_i < 0.22) &
                          (i_z > -0.21) & (i_z < 0.25)]

    #----------------------------------------------------------------------
    # get magnitudes and colors; split into train and test sets

    mags_rr = np.vstack([rrlyrae[f + 'mag'] for f in 'ugriz'])
    colors_rr = mags_rr[:-1] - mags_rr[1:]

    mags_st = np.vstack([standards['mmu_' + f] for f in 'ugriz'])
    colors_st = mags_st[:-1] - mags_st[1:]

    # stack the two sets of colors together
    X = np.vstack((colors_st.T, colors_rr.T))
    y = np.zeros(X.shape[0])
    y[-colors_rr.shape[1]:] = 1

    return X, y


def split_samples(X, y, rseed=0, training_fraction=0.75):
    """split samples into training and test sets"""
    np.random.seed(0)
    indices = np.arange(len(y))
    np.random.shuffle(indices)

    N_train = int(training_fraction * len(y))

    X_train = X[indices[:N_train]]
    X_test = X[indices[N_train:]]

    y_train = y[indices[:N_train]]
    y_test = y[indices[N_train:]]

    return X_train, X_test, y_train, y_test

#----------------------------------------------------------------------
# get data and split into training & testing sets
X, y = load_rrlyrae_data()

# SVM takes several minutes to run, and is order[N^2]
#  truncating the dataset can be useful for experimentation.
#X = X[::10]
#y = y[::10]

X_train, X_test, y_train, y_test = split_samples(X, y)

N_tot = len(y)
N_st = np.sum(y == 0)
N_rr = N_tot - N_st
N_train = len(y_train)
N_test = len(y_test)
N_plot = 5000 + N_rr

#----------------------------------------------------------------------
# Fit Decision tree
Ncolors = np.arange(1, X.shape[1] + 1)

classifiers = []
predictions = []
Ncolors = np.arange(1, X.shape[1] + 1)
depths = [5, 20]

for depth in depths:
    classifiers.append([])
    predictions.append([])
    for nc in Ncolors:
        clf = RandomForestClassifier(random_state=0, max_depth=depth,
                                     criterion='entropy')
        clf.fit(X_train[:, :nc], y_train)
        y_pred = clf.predict(X_test[:, :nc])

        classifiers[-1].append(clf)
        predictions[-1].append(y_pred)

predictions = np.array(predictions)

#----------------------------------------------------------------------
# compute completeness and contamination
#
matches = (predictions == y_test)

tp = np.sum(matches & (y_test == 1), -1)
tn = np.sum(matches & (y_test == 0), -1)
fp = np.sum(~matches & (y_test == 0), -1)
fn = np.sum(~matches & (y_test == 1), -1)

completeness = tp * 1. / (tp + fn)
contamination = fp * 1. / (tp + fp)

completeness[np.isnan(completeness)] = 0
contamination[np.isnan(contamination)] = 0

print("completeness", completeness)
print("contamination", contamination)

#----------------------------------------------------------------------
# plot the results
plt.figure(figsize=(5, 2.5))
plt.subplots_adjust(bottom=0.15, top=0.95, hspace=0.0,
                    left=0.1, right=0.95, wspace=0.2)

ax = plt.subplot(121)
plt.scatter(X[-N_plot:, 0], X[-N_plot:, 1], c=y[-N_plot:],
            s=4, lw=0, cmap=plt.cm.binary, zorder=2)
plt.clim(-0.5, 1)

clf = classifiers[1][1]
xlim = (0.7, 1.35)
ylim = (-0.15, 0.4)

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 101),
                     np.linspace(ylim[0], ylim[1], 101))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# smooth the boundary
#from scipy.ndimage import gaussian_filter
#Z = gaussian_filter(Z, 2)

plt.contour(xx, yy, Z, [0.5], colors='k')

plt.xlim(xlim)
plt.ylim(ylim)

plt.xlabel('$u-g$')
plt.ylabel('$g-r$')

plt.text(0.02, 0.02, "depth = %i" % depths[1],
         transform=ax.transAxes)

ax = plt.subplot(222)
plt.plot(Ncolors, completeness[0], 'o-k', ms=6, label="depth=%i" % depths[0])
plt.plot(Ncolors, completeness[1], '^--k', ms=6, label="depth=%i" % depths[1])

ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
ax.xaxis.set_major_formatter(plt.NullFormatter())

plt.ylabel('completeness')
plt.xlim(0.5, 4.5)
plt.ylim(-0.1, 1.1)
plt.grid(True)

ax = plt.subplot(224)
plt.plot(Ncolors, contamination[0], 'o-k', ms=6, label="depth=%i" % depths[0])
plt.plot(Ncolors, contamination[1], '^--k', ms=6, label="depth=%i" % depths[1])
plt.legend(loc='lower right',
           bbox_to_anchor=(1.0, 0.79))

ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%i'))
plt.xlabel('N colors')
plt.ylabel('contamination')
plt.xlim(0.5, 4.5)
plt.ylim(-0.1, 1.1)
plt.grid(True)

plt.show()
