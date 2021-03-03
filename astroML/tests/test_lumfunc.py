import numpy as np
from astroML.lumfunc import Cminus


def test_cminus_nans():
    # Regression test for https://github.com/astroML/astroML/issues/234

    x = [10.02, 10.00]
    y = [14.97, 14.99]
    xmax = [10.03, 10.01]
    ymax = [14.98, 15.00]

    assert np.isfinite(np.sum(Cminus(x, y, xmax, ymax)))
