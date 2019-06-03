import numpy as np
from numpy.testing import assert_allclose
import pytest
from astroML.linear_model import NadarayaWatson


def test_NW_simple():
    X = np.arange(11.)
    y = X + 1
    dy = 1

    # by symmetry, NW regression should get these exactly correct
    Xfit = np.array([4, 5, 6])[:, None]
    y_true = np.ravel(Xfit + 1)

    clf = NadarayaWatson(h=0.5).fit(X[:, None], y, dy)
    y_fit = clf.predict(Xfit)

    assert_allclose(y_fit, y_true)


def test_NW_simple_laplacian_kernel():
    X = np.arange(11.)
    y = X + 1
    dy = 1

    # by symmetry, NW regression should get these exactly correct
    Xfit = np.array([4, 5, 6])[:, None]
    y_true = np.ravel(Xfit + 1)

    kwargs = {'gamma': 10.}
    clf = NadarayaWatson(kernel='laplacian', **kwargs).fit(X[:, None], y, dy)
    y_fit = clf.predict(Xfit)

    assert_allclose(y_fit, y_true)


def test_X_invalid_shape_exception():
    X = np.arange(11.)
    y = X + 1
    dy = 1

    clf = NadarayaWatson(h=0.5).fit(X[:, None], y, dy)

    # not valid Xfit.shape[1], should raise an exception
    Xfit = np.array([[4, 5, 6], [1, 2, 3]])
    y_true = np.ravel(Xfit + 1)

    with pytest.raises(Exception) as e:
        y_fit = clf.predict(Xfit)

    assert str(e.value) == "dimensions of X do not match training dimension"

    # not valid Xfit.shape[1], should raise an exception
    Xfit = np.array([4, 5, 6])
    y_true = np.ravel(Xfit + 1)

    with pytest.raises(Exception) as e:
        y_fit = clf.predict(Xfit)

    assert str(e.value) == "X must be two-dimensional"