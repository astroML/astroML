import numpy as np
from numpy.testing import assert_allclose
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
