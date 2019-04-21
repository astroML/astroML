"""Tests of the GMM Bayes classifier"""
import numpy as np
from numpy.testing import assert_allclose
import pytest
import warnings
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


def test_incompatible_shapes_exception():
    X = np.random.normal(0, 1, size=(100, 2))
    y = np.zeros(99)

    ncm = 1
    clf = GMMBayes(ncm)

    with pytest.raises(Exception) as e:
        assert clf.fit(X, y)

    assert str(e.value) == "X and y have incompatible shapes"


def test_incompatible_number_of_components_exception():
    X = np.random.normal(0, 1, size=(100, 2))
    y = np.zeros(100)

    ncm = [1, 2, 3]
    clf = GMMBayes(ncm)

    with pytest.raises(Exception) as e:
        assert clf.fit(X, y)

    assert str(e.value) == ("n_components must be compatible with "
                             "the number of classes")


def test_too_many_components_warning():
    X = np.random.normal(0, 1, size=(3, 2))
    y = np.zeros(3)

    ncm = 5
    clf = GMMBayes(ncm)

    with pytest.warns(UserWarning, match="Expected n_samples >= "
                                         "n_components but got "):
        clf.fit(X, y)