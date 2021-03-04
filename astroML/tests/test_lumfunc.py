import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from astroML.lumfunc import _sorted_interpolate, Cminus, binned_Cminus, bootstrap_Cminus
from scipy import stats

def test_sorted_interp():
    x = np.arange(0, 10)
    y1 = 3*x + 4
    y2 = 0.5*x**2 + 8.0*x - 6.0

    xtest = np.array([2.5, 4.5, 6.5, 8.5])
    ytest1 = _sorted_interpolate(x, y1, xtest)
    ytest2 = _sorted_interpolate(x, y2, xtest)

    assert_equal(ytest1, 3*xtest + 4)
    assert_almost_equal(ytest2, 0.5*xtest**2 + 8.0*xtest - 6.0, decimal=1)

def test_bootstrap_Cminus():
    N = 10000
    np.random.seed(42)

    # Define the input distributions for x and y
    x_pdf = stats.truncnorm(-2, 1, 0.66666, 0.33333)
    y_pdf = stats.truncnorm(-1, 2, 0.33333, 0.33333)

    x = x_pdf.rvs(N)
    y = y_pdf.rvs(N)

    # define the truncation: we'll design this to be symmetric
    # so that xmax(y) = max_func(y)
    # and ymax(x) = max_func(x)
    max_func = lambda t: 1. / (0.5 + t) - 0.5

    xmax = max_func(y)
    xmax[xmax > 1] = 1  # cutoff at x=1

    ymax = max_func(x)
    ymax[ymax > 1] = 1  # cutoff at y=1

    # truncate the data
    flag = (x < xmax) & (y < ymax)
    x = x[flag]
    y = y[flag]
    xmax = xmax[flag]
    ymax = ymax[flag]

    x_fit = np.linspace(0, 1, 11)
    y_fit = np.linspace(0, 1, 11)

    Nbootstraps= 10

    x_dist, x_unc, y_dist, y_unc, cuml_x, cuml_y = bootstrap_Cminus(x, y, xmax, ymax, x_fit, y_fit, normalize=True, return_cumulative=True, Nbootstraps=Nbootstraps)
    x_mid = 0.5 * (x_fit[1:] + x_fit[:-1])
    y_mid = 0.5 * (y_fit[1:] + y_fit[:-1])

    x_pdf_points = x_pdf.pdf(x_mid)
    x_cdf_points = x_pdf.cdf(x_mid)
    y_pdf_points = y_pdf.pdf(y_mid)
    y_cdf_points = y_pdf.cdf(y_mid)
    
    assert_almost_equal(x_dist*Nbootstraps, x_pdf_points, decimal=1)
    assert_almost_equal(y_dist*Nbootstraps, y_pdf_points, decimal=1)
    assert_almost_equal(cuml_x/cuml_x[-1], x_cdf_points, decimal=1)
    assert_almost_equal(cuml_y/cuml_y[-1], y_cdf_points, decimal=1)

