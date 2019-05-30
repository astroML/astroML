import numpy as np
from numpy.testing import assert_allclose, run_module_suite

from astroML.resample import bootstrap, jackknife
from astroML.stats import mean_sigma


def test_jackknife_results():
    np.random.seed(0)
    x = np.random.normal(0, 1, 100)
    mu1, sig1 = jackknife(x, np.mean, kwargs=dict(axis=1))
    mu2, sig2 = jackknife(x, np.std, kwargs=dict(axis=1))

    assert_allclose([mu1, sig1, mu2, sig2],
                    [0.0598080155345, 0.100288031685,
                     1.01510470168, 0.0649020337599])


def test_jackknife_multiple():
    np.random.seed(0)
    x = np.random.normal(0, 1, 100)

    mu1, sig1 = jackknife(x, np.mean, kwargs=dict(axis=1))
    mu2, sig2 = jackknife(x, np.std, kwargs=dict(axis=1))

    res = jackknife(x, mean_sigma, kwargs=dict(axis=1))

    assert_allclose(res[0], (mu1, sig1))
    assert_allclose(res[1], (mu2, sig2))


def test_bootstrap_results():
    np.random.seed(0)
    x = np.random.normal(0, 1, 100)
    distribution = bootstrap(x, 100, np.mean, kwargs=dict(axis=1),
                             random_state=0)

    mu, sigma = mean_sigma(distribution)

    assert_allclose([mu, sigma], [0.08139846, 0.10465327])


def test_bootstrap_multiple():
    np.random.seed(0)
    x = np.random.normal(0, 1, 100)

    dist_mean = bootstrap(x, 100, np.mean, kwargs=dict(axis=1),
                          random_state=0)
    dist_std = bootstrap(x, 100, np.std, kwargs=dict(axis=1),
                         random_state=0)
    res = bootstrap(x, 100, mean_sigma, kwargs=dict(axis=1),
                    random_state=0)

    assert_allclose(res[0], dist_mean)
    assert_allclose(res[1], dist_std)

def test_bootstrap_covar():
    np.random.seed(0)
    mean = [0.,0.]
    covar = [[10.,3.],[3.,20.]]
    x = np.random.multivariate_normal(mean, covar, 1000)

    dist_cov = bootstrap(x, 10000, np.cov, kwargs=dict(rowvar=0), random_state=0)
    assert_allclose(covar[0][0], dist_cov[0][0], atol=2.*0.4)


def test_bootstrap_pass_indices():
    np.random.seed(0)
    x = np.random.normal(0, 1, 100)

    dist1 = bootstrap(x, 100, np.mean,
                      kwargs=dict(axis=1), random_state=0)
    dist2 = bootstrap(x, 100, lambda i: np.mean(x[i], axis=1),
                      pass_indices=True, random_state=0)

    assert_allclose(dist1, dist2)


def test_jackknife_pass_indices():
    np.random.seed(0)
    x = np.random.normal(0, 1, 100)

    res1 = jackknife(x, np.mean,
                     kwargs=dict(axis=1))
    res2 = jackknife(x, lambda i: np.mean(x[i], axis=1),
                     pass_indices=True)

    assert_allclose(res1, res2)


if __name__ == '__main__':
    run_module_suite()
