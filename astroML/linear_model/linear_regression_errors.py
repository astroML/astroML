import numpy as np

try:
    import pymc3 as pm
    import theano.tensor as tt
except ImportError:
    raise ImportError('LinearRegressionwithErrors requires PyMC3 to be installed.')


from astroML.linear_model import LinearRegression


__all__ = ['LinearRegressionwithErrors']


class LinearRegressionwithErrors(LinearRegression):

    def __init__(self, fit_intercept=False, regularization='none', kwds=None):
        super().__init__(fit_intercept, regularization, kwds)

    def fit(self, X, y, y_error=1, x_error=None, *,
            sample_kwargs={'draws': 1000, 'target_accept': 0.9}):

        kwds = {}
        if self.kwds is not None:
            kwds.update(self.kwds)
        kwds['fit_intercept'] = False
        model = self._choose_regressor()
        self.clf_ = model(**kwds)

        self.fit_intercept = False

        if x_error is not None:
            x_error = np.atleast_2d(x_error)
        with pm.Model():
            # slope and intercept of eta-ksi relation
            slope = pm.Flat('slope', shape=(X.shape[0], ))
            inter = pm.Flat('inter')

            # intrinsic scatter of eta-ksi relation
            int_std = pm.HalfFlat('int_std')
            # standard deviation of Gaussian that ksi are drawn from (assumed mean zero)
            tau = pm.HalfFlat('tau', shape=(X.shape[0],))
            # intrinsic ksi
            mu = pm.Normal('mu', mu=0, sd=tau, shape=(X.shape[0],))

            # Some wizzarding with the dimensions all around.
            ksi = pm.Normal('ksi', mu=mu, tau=tau, shape=X.T.shape)

            # intrinsic eta-ksi linear relation + intrinsic scatter
            eta = pm.Normal('eta', mu=(tt.dot(slope.T, ksi.T) + inter),
                            sd=int_std, shape=y.shape)

            # observed xi, yi
            x = pm.Normal('xi', mu=ksi.T, sd=x_error, observed=X, shape=X.shape)
            y = pm.Normal('yi', mu=eta, sd=y_error, observed=y, shape=y.shape)

            self.trace = pm.sample(**sample_kwargs)

            # TODO big: make it optional to choose a way to define best

            # TODO quick: use np.histogramdd
            H2D, bins1, bins2 = np.histogram2d(self.trace['slope'][:, 0],
                                               self.trace['inter'], bins=50)

            w = np.where(H2D == H2D.max())

            # choose the maximum posterior slope and intercept
            slope_best = bins1[w[0][0]]
            intercept_best = bins2[w[1][0]]
            self.clf_.coef_ = np.array([intercept_best, slope_best])

        return self
