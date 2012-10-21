"""
Clustering of LINEAR data
-------------------------
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt

from sklearn.mixture import GMM

from astroML.decorators import pickle_results
from astroML.datasets import fetch_LINEAR_geneva

#------------------------------------------------------------
# Get the Geneva periods data
data = fetch_LINEAR_geneva()

#----------------------------------------------------------------------
# compute Gaussian Mixture models

filetemplate = 'gmm_res_%i_%i.pkl'
attributes = [('gi', 'logP'),
              ('ug', 'gi', 'iK', 'JK', 'logP', 'amp', 'skew')]
labels = ['$u-g$', '$g-i$', '$i-K$', '$J-K$',
          r'$\log(P)$', 'amplitude', 'skew']
components = np.arange(1, 21)

#------------------------------------------------------------
# Create attribute arrays
Xarrays = []
for attr in attributes:
    Xarrays.append(np.vstack([data[a] for a in attr]).T)


#------------------------------------------------------------
# Compute the results (and save to pickle file)
@pickle_results('LINEAR_clustering.pkl')
def compute_GMM_results(components, attributes):
    clfs = []

    for attr, X in zip(attributes, Xarrays):
        print attr, X.shape
        clfs_i = []

        for comp in components:
            print "  - %i component fit" % comp
            clf = GMM(comp, covariance_type='full',
                      random_state=0, n_iter=500)
            clf.fit(X)
            clfs_i.append(clf)

            if not clf.converged_:
                print "           NOT CONVERGED!"

        clfs.append(clfs_i)
    return clfs

clfs = compute_GMM_results(components, attributes)

#------------------------------------------------------------
# Plot the results
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i in range(2):
    # Grab the best classifier, based on the BIC
    X = Xarrays[i]
    BIC = [c.bic(X) for c in clfs[i]]
    i_best = np.argmin(BIC)
    clf = clfs[i][i_best]
    n_components = clf.n_components

    # Predict the cluster labels with the classifier
    c = clf.predict(X)
    classes = np.unique(c)

    # sort the cluster by normalized density of points
    counts = np.sum(c == classes[:, None], 1)
    size = np.array([np.linalg.det(C) for C in clf.covars_])
    weights = clf.weights_
    density = counts * 1. / weights / size
    isort = np.argsort(density)[::-1]

    # find statistics of the top 5 clusters
    means = []
    stdevs = []
    counts = []

    names = data.dtype.names[2:]
    i_logP = names.index('logP')

    for j in range(5):
        flag = (c == isort[j])
        counts.append(np.sum(flag))
        means.append([np.mean(data[n][flag]) for n in names])
        stdevs.append([data[n][flag].std() for n in names])

    counts = np.array(counts)
    means = np.array(means)
    stdevs = np.array(stdevs)

    # define colors based on median of logP
    j_ordered = np.argsort(means[:, i_logP])[::-1]
    color = np.zeros(c.shape)
    for j in range(5):
        flag = (c == isort[j_ordered[j]])
        color[flag] = j + 1

    # separate into foureground and background
    back = (color == 0)
    fore = ~back

    # Plot the resulting clusters
    ax1 = fig.add_subplot(221 + 2 * i)
    ax1.scatter(data['gi'][back], data['logP'][back],
                c='gray', s=4, linewidths=0)
    ax1.scatter(data['gi'][fore], data['logP'][fore],
                c=color[fore], s=4, linewidths=0)

    ax1.set_ylabel(r'$\log(P)$')

    ax2 = plt.subplot(222 + 2 * i)
    ax2.scatter(data['amp'][back], data['logP'][back],
                c='gray', s=4, lw=0)
    ax2.scatter(data['amp'][fore], data['logP'][fore],
                c=color[fore], s=4, lw=0)

    #------------------------------
    # set axis limits
    ax1.set_xlim(-0.6, 2.1)
    ax2.set_xlim(0.1, 1.5)
    ax1.set_ylim(-1.5, 0.5)
    ax2.set_ylim(-1.5, 0.5)

    ax2.yaxis.set_major_formatter(plt.NullFormatter())
    if i == 0:
        ax1.xaxis.set_major_formatter(plt.NullFormatter())
        ax2.xaxis.set_major_formatter(plt.NullFormatter())
    else:
        ax1.set_xlabel(r'$g-i$')
        ax2.set_xlabel(r'$A$')

    #------------------------------
    # print table of means and medians directly to LaTeX format
    print r"\begin{tabular}{|l|lllllll|}"
    print r"   \hline"
    for j in range(len(names)):
        print '   &', labels[j],
    print r"\\"
    print r"   \hline"

    for j in range(5):
        print "   %i " % (j + 1),
        for k in range(len(names)):
            print " & $%.2f \pm %.2f$ " % (means[j, k], stdevs[j, k]),
        print r"\\"

    print r"\hline"
    print r"\end{tabular}"

#------------------------------------------------------------
# Second figure
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(left=0.11, right=0.95, wspace=0.3)

attrs = ['skew', 'ug', 'iK', 'JK']
labels = ['skew', '$u-g$', '$i-K$', '$J-K$']
ylims = [(-1.8, 2.2), (0.6, 2.9), (0.1, 2.6), (-0.2, 1.2)]

for i in range(4):
    ax = fig.add_subplot(221 + i)
    ax.scatter(data['gi'][back], data[attrs[i]][back],
               c='gray', s=4, lw=0)
    ax.scatter(data['gi'][fore], data[attrs[i]][fore],
               c=color[fore], s=4, lw=0)
    ax.set_xlabel('$g-i$')
    ax.set_ylabel(labels[i])

    ax.set_xlim(-0.6, 2.1)
    ax.set_ylim(ylims[i])

plt.show()
