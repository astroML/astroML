"""Tests of the GMM Bayes classifier"""
import numpy as np
from numpy.testing import assert_allclose
from astroML.classification import GMMBayes


def test_gmm1d():
    x1 = np.random.normal(0, 1, size=100)
    x2 = np.random.normal(10, 1, size=100)
    X = np.concatenate((x1, x2)).reshape((200, 1))
    y = np.zeros(200)
    y[100:] = 1

    ncm = 1
    clf = GMMBayes(ncm)
    clf.fit(X, y)

    predicted = clf.predict(X)
    assert_allclose(y, predicted)


def test_gmm2d():
    x1 = np.random.normal(0, 1, size=(100, 2))
    x2 = np.random.normal(10, 1, size=(100, 2))
    X = np.vstack((x1, x2))
    y = np.zeros(200)
    y[100:] = 1

    for ncm in (1, 2, 3):
        clf = GMMBayes(ncm)
        clf.fit(X, y)

        predicted = clf.predict(X)
        assert_allclose(y, predicted)
