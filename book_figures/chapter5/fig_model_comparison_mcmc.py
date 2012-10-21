"""
MCMC Model Comparison
---------------------
This example uses Bayesian MCMC to select between two models for a dataset.
The dataset is a mixture of two gaussian distributions, and two MCMC models
are fit: one with a single gaussian, one with two gaussians of different
widths.
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt

from scipy.special import gamma
from scipy.stats import norm
from sklearn.neighbors import BallTree
from astroML.density_estimation import GaussianMixture1D
from astroML.plotting import plot_mcmc
from astroML.decorators import pickle_results

# hack to fix an import issue in older versions of pymc
import scipy
scipy.derivative = scipy.misc.derivative
import pymc


def get_logp(S, model):
    """compute log(p) given a pyMC model"""
    M = pymc.MAP(model)
    traces = np.array([S.trace(s)[:] for s in S.stochastics])
    logp = np.zeros(traces.shape[1])
    for i in range(len(logp)):
        logp[i] = -M.func(traces[:, i])
    return logp


def estimate_bayes_factor(traces, logp, r=0.05, return_list=False):
    """Estimate the bayes factor using the local density of points"""
    D, N = traces.shape

    # compute volume of a D-dimensional sphere of radius r
    Vr = np.pi ** (0.5 * D) / gamma(0.5 * D + 1) * (r ** D)

    # use neighbor count within r as a density estimator
    bt = BallTree(traces.T)
    count = bt.query_radius(traces.T, r=r, count_only=True)

    BF = logp + np.log(N) + np.log(Vr) - np.log(count)

    if return_list:
        return BF
    else:
        p25, p50, p75 = np.percentile(BF, [25, 50, 75])
        return p50, 0.7413 * (p75 - p25)


#------------------------------------------------------------
# Generate the data
mu1_in = 0
sigma1_in = 0.3
mu2_in = 1
sigma2_in = 1
ratio_in = 1.5
N = 200

np.random.seed(10)
gm = GaussianMixture1D([mu1_in, mu2_in],
                       [sigma1_in, sigma2_in],
                       [ratio_in, 1])
x_sample = gm.sample(N)

#------------------------------------------------------------
# Set up pyMC model: single gaussian
#  2 parameters: (mu, sigma)
M1_mu = pymc.Uniform('M1_mu', -5, 5, value=0)
M1_log_sigma = pymc.Uniform('M1_log_sigma', -10, 10, value=0)


@pymc.deterministic
def M1_sigma(M1_log_sigma=M1_log_sigma):
    return np.exp(M1_log_sigma)


@pymc.deterministic
def M1_tau(M1_sigma=M1_sigma):
    return 1. / M1_sigma ** 2

M1 = pymc.Normal('M1', M1_mu, M1_tau, observed=True, value=x_sample)
model1 = dict(M1_mu=M1_mu, M1_log_sigma=M1_log_sigma,
              M1_sigma=M1_sigma,
              M1_tau=M1_tau, M1=M1)


#------------------------------------------------------------
# Set up pyMC model: double gaussian
#  5 parameters: (mu1, mu2, sigma1, sigma2, ratio)

def doublegauss_like(x, mu1, mu2, sigma1, sigma2, ratio):
    """log-likelihood for double gaussian"""
    r1 = ratio / (1. + ratio)
    r2 = 1 - r1
    L = r1 * norm(mu1, sigma1).pdf(x) + r2 * norm(mu2, sigma2).pdf(x)
    L[L == 0] = 1E-16  # prevent divide-by-zero error
    logL = np.log(L).sum()
    if np.isinf(logL):
        raise pymc.ZeroProbability
    else:
        return logL


def rdoublegauss(mu1, mu2, sigma1, sigma2, ratio, size=None):
    """random variable from double gaussian"""
    r1 = ratio / (1. + ratio)
    r2 = 1 - r1
    R = np.asarray(np.random.random(size))

    Rshape = R.shape
    R = np.atleast1d(R)

    mask1 = (R < r1)
    mask2 = ~mask1
    N1 = mask1.sum()
    N2 = R.size - N1

    R[mask1] = norm(mu1, sigma1).rvs(N1)
    R[mask2] = norm(mu2, sigma2).rvs(N2)

    return R.reshape(Rshape)

DoubleGauss = pymc.stochastic_from_dist('doublegauss',
                                        logp=doublegauss_like,
                                        random=rdoublegauss,
                                        dtype=np.float,
                                        mv=True)

# set up our Stochastic variables, mu1, mu2, sigma1, sigma2, ratio
M2_mu1 = pymc.Uniform('M2_mu1', -5, 5, value=0)
M2_mu2 = pymc.Uniform('M2_mu2', -5, 5, value=1)

M2_log_sigma1 = pymc.Uniform('M2_log_sigma1', -10, 10, value=0)
M2_log_sigma2 = pymc.Uniform('M2_log_sigma2', -10, 10, value=0)


@pymc.deterministic
def M2_sigma1(M2_log_sigma1=M2_log_sigma1):
    return np.exp(M2_log_sigma1)


@pymc.deterministic
def M2_sigma2(M2_log_sigma2=M2_log_sigma2):
    return np.exp(M2_log_sigma2)

M2_ratio = pymc.Uniform('M2_ratio', 1E-3, 1E3, value=1)

M2 = DoubleGauss('M2', M2_mu1, M2_mu2, M2_sigma1, M2_sigma2, M2_ratio,
                 observed=True, value=x_sample)

model2 = dict(M2_mu1=M2_mu1, M2_mu2=M2_mu2,
              M2_log_sigma1=M2_log_sigma1, M2_log_sigma2=M2_log_sigma2,
              M2_sigma1=M2_sigma1, M2_sigma2=M2_sigma2,
              M2_ratio=M2_ratio, M2=M2)


#------------------------------------------------------------
# Set up MCMC sampling
#   We'll save the results, because the calculation takes a while
@pickle_results('model_comparison_traces.pkl')
def compute_MCMC_models(Niter=10000, burn=1000, rseed=0):
    pymc.numpy.random.seed(rseed)

    S1 = pymc.MCMC(model1)
    S1.sample(iter=Niter, burn=burn)
    trace1 = np.vstack([S1.trace('M1_mu')[:],
                        S1.trace('M1_sigma')[:]])
    logp1 = get_logp(S1, model1)

    S2 = pymc.MCMC(model2)
    S2.sample(iter=Niter, burn=burn)
    trace2 = np.vstack([S2.trace('M2_mu1')[:],
                        S2.trace('M2_mu2')[:],
                        S2.trace('M2_sigma1')[:],
                        S2.trace('M2_sigma2')[:],
                        S2.trace('M2_ratio')[:]])
    logp2 = get_logp(S2, model2)

    return trace1, logp1, trace2, logp2

trace1, logp1, trace2, logp2 = compute_MCMC_models()


#------------------------------------------------------------
# Compute Odds ratio with density estimation technique
BF1, dBF1 = estimate_bayes_factor(trace1, logp1, r=0.02)
BF2, dBF2 = estimate_bayes_factor(trace2, logp2, r=0.05)


#------------------------------------------------------------
# Plot the results
fig = plt.figure(figsize=(10, 10))

labels = [r'$\mu_1$',
          r'$\mu_2$',
          r'$\sigma_1$',
          r'$\sigma_2$',
          r'${\rm ratio}$']

true_values = [mu1_in,
               mu2_in,
               sigma1_in,
               sigma2_in,
               ratio_in]

limits = [(-0.24, 0.12),
          (0.55, 1.75),
          (0.15, 0.45),
          (0.55, 1.3),
          (0.25, 2.1)]

# we assume mu1 < mu2, but the results may be switched
#  due to the symmetry of the problem.  If so, switch back
if np.median(trace2[0]) > np.median(trace2[1]):
    trace2 = trace2[[1, 0, 3, 2, 4], :]
    N2_norm_mu = N2.mu[N2.M2_mu2, N2.M2_mu1,
                       N2.M2_sigma2, N2.M2_sigma1, N2.M2_ratio]
    N2_norm_Sig = N2.C[N2.M2_mu2, N2.M2_mu1,
                       N2.M2_sigma2, N2.M2_sigma1, N2.M2_ratio]

# Plot the simple 2-component model
ax, = plot_mcmc(trace1, fig=fig, bounds=[0.6, 0.6, 0.95, 0.95],
                limits=[(0.3, 0.8), (0.75, 1.15)],
                labels=[r'$\mu$', r'$\sigma$'], colors='k', linewidths=2)

ax.text(0.03, 0.97, "Single Gaussian fit", va='top', ha='left',
        transform=ax.transAxes, fontsize=14)

# Plot the 5-component model
ax_list = plot_mcmc(trace2, limits=limits, labels=labels,
                    true_values=true_values, fig=fig,
                    colors='k', linewidths=2)
for ax in ax_list:
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(plt.MaxNLocator(4))

plt.show()
