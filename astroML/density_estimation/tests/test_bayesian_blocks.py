import numpy as np
from  numpy.testing import assert_allclose, assert_
from astroML.density_estimation import histogram_bayesian_blocks

def test_single_change_point():
    # make sure the correct change point is found for a simple distribution
    np.random.seed(0)
    x = np.concatenate([np.random.random(1000),
                        1 + np.random.random(2000)])

    counts, bins = histogram_bayesian_blocks(x)
    change_point = bins[1]

    assert_allclose(change_point, 1, rtol=0.01)
