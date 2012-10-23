"""
Decision Tree for Star/Galaxy Classification
--------------------------------------------

This uses a decision tree to classify photometric objects as either
stars or galaxies.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com

import numpy as np
from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from astroML.datasets import fetch_imaging_sample


def get_x_position(level, n_levels, xmin=0.01, xmax=1):
    dx = (xmax - xmin) / (n_levels - 1.)
    return xmin + level * dx


def get_y_position(level, j, ymin=0, ymax=1):
    n = 2 ** level
    dy = (ymax - ymin) * 1. / n
    return ymin + (j + 0.5) * dy


def draw_connections(x_positions, y_positions, children, i, linestyle='-k'):
    for c, y in zip(children, y_positions):
        if c == -1:
            continue
        plt.plot(x_positions[i - 1:i + 1], [y, y], linestyle, lw=1)

    for j in range(0, 2 ** i, 2):
        if children[j] == -1 or children[j + 1] == -1:
            continue
        plt.plot(2 * [x_positions[i - 1]], y_positions[j:j + 2],
                 linestyle, lw=1)


def visualize_tree(T, data, classes, labels=None, levels=5,
                   ymin=0, ymax=1, xmin=0, xmax=1):
    # to visualize the tree, we essentially need to re-build it: it doesn't
    # store the list of points at each node.

    # get tree aspects
    T_children = T.tree_.children
    T_nsamples = T.tree_.n_samples
    T_feature = T.tree_.feature
    T_threshold = T.tree_.threshold

    x_positions = get_x_position(np.arange(levels + 1), levels)
    node_list = np.array([0])
    new_data_masks = [np.ones(data.shape[0], dtype=bool)]

    for i in range(levels):
        y_positions = get_y_position(i, np.arange(2 ** i))

        mask = (node_list != -1)

        # plot the positions of the nodes
        plt.plot(x_positions[i] * np.ones(2 ** i)[mask],
                 y_positions[mask], 'ok')

        data_masks = new_data_masks
        new_data_masks = []

        # list node info
        for j in range(2 ** i):
            if node_list[j] == -1:
                new_data_masks += [None, None]
                continue
            ind = node_list[j]

            # get masks of children
            split_mask = (data[:, T_feature[ind]] < T_threshold[ind])
            new_data_masks.append(np.logical_and(data_masks[j], split_mask))
            new_data_masks.append(np.logical_and(data_masks[j], ~split_mask))

            n_stars = np.sum(classes[data_masks[j]] == 3)
            n_gals = np.sum(classes[data_masks[j]] == 6)

            text = "$%i\ /\ %i$" % (n_stars, n_gals)

            # assure that we're doing this correctly
            assert (n_stars + n_gals == T_nsamples[ind])

            # check if node is a leaf
            if n_stars == 0:
                text += "\n" + r"$\rm(galaxies)$"
            elif n_gals == 0:
                text += "\n" + r"$\rm(stars)$"
            else:
                text += "\n" + r"$\rm split\ on$ %s" % labels[T_feature[ind]]

            if i < 4:
                fontsize = 12
            else:
                fontsize = 10

            plt.text(x_positions[i], y_positions[j], text,
                     ha='center', va='center',
                     fontsize=fontsize,
                     bbox=dict(boxstyle='round', ec='k', fc='w'))

        # draw lines connecting nodes to parents
        if i > 0:
            draw_connections(x_positions, y_positions, node_list, i, '-k')

        # get next set of nodes
        node_list = np.concatenate(list(T_children[node_list]))

    # draw dotted line for last level
    y_positions = get_y_position(levels, np.arange(2 ** levels))
    draw_connections(x_positions, y_positions, node_list, levels, ':k')

    # set suitable axes limits
    dx = 0.1 * (xmax - xmin)
    dy = 0.02 * (xmax - xmin)
    plt.xlim(xmin - dx, xmax + 2 * dx)
    plt.ylim(ymin - dy, ymax + dy)

data = fetch_imaging_sample()

mag = np.vstack([data[f + 'Raw'] for f in 'ugriz']
                + [data[f + 'RawPSF'] for f in 'ugriz']).T

label = data['type']

mag_train = mag[::40]
mag_test = mag[1::2]

label_train = label[::40]
label_test = label[1::2]

clf = DecisionTreeClassifier(compute_importances=True,
                             random_state=0, criterion='entropy')
clf.fit(mag_train, label_train)

label_out = clf.predict(mag_test)

eq = (label_out == label_test)

#--------------------------------------------------
# compute statistics of cross-validation
tot_neg = np.sum(label_test == 3)
tot_pos = np.sum(label_test == 6)

fn = np.sum((label_test == 6) & (label_out == 3))
tn = np.sum((label_test == 3) & (label_out == 3))

fp = np.sum((label_test == 3) & (label_out == 6))
tp = np.sum((label_test == 6) & (label_out == 6))

print "----------------------------------------------------------------"
print ("partial training set: (%i stars, %i galaxies)"
       % (np.sum(label_train == 3), np.sum(label_train == 6)))
print "positive = galaxy, negative = star"
print "false positives: %i (%.1f%%)" % (fp, fp * 100. / (fp + tp))
print "false negatives: %i (%.1f%%)" % (fn, fn * 100. / (fn + tn))

plt.figure(figsize=(8, 10), facecolor='w')
plt.axes([0, 0, 1, 1], xticks=[], yticks=[], frameon=False)
visualize_tree(clf, mag_train, label_train,
               labels=([r'$\rm %s\_mod$' % f for f in 'ugriz'] +
                       [r'$\rm %s\_psf$' % f for f in 'ugriz']))

plt.text(0.1, 0.95, (" Numbers are\n"
                     " star count / galaxy count\n"
                     " in each node"),
         ha='center', va='center',
         fontsize=12,
         bbox=dict(boxstyle='round', ec='k', fc='w'))

plt.text(-0.08, 0.01, ("Training Set Size:\n"
                       "  %i objects\n\n"
                       "Cross-Validation, with\n"
                       "  %i galaxies (positive)\n"
                       "  %i stars (negative)\n"
                       "  false positives: %i (%.1f%%)\n"
                       "  false negatives: %i (%.1f%%)"
                       % (len(label_train), tot_pos, tot_neg,
                          fp, fp * 100. / (tp + fp),
                          fn, fn * 100. / (tn + fn))),
         fontsize=12, ha='left', va='bottom')

#--------------------------------------------------
# compute statistics for a larger training set

mag_train = mag[::2]
mag_test = mag[1::2]

label_train = label[::2]
label_test = label[1::2]

clf = DecisionTreeClassifier(compute_importances=True,
                             random_state=0, criterion='entropy')
clf.fit(mag_train, label_train)

label_out = clf.predict(mag_test)

tot_neg = np.sum(label_test == 3)
tot_pos = np.sum(label_test == 6)

fn = np.sum((label_test == 6) & (label_out == 3))
tn = np.sum((label_test == 3) & (label_out == 3))

fp = np.sum((label_test == 3) & (label_out == 6))
tp = np.sum((label_test == 6) & (label_out == 6))

print "----------------------------------------------------------------"
print ("full training set: (%i stars, %i galaxies)"
       % (np.sum(label_train == 3), np.sum(label_train == 6)))
print "positive = galaxy, negative = star"
print "false positives: %i (%.1f%%)" % (fp, fp * 100. / (fp + tp))
print "false negatives: %i (%.1f%%)" % (fn, fn * 100. / (fn + tn))

plt.show()
