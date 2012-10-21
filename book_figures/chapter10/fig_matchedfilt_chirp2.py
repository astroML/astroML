"""
Matched Filter Burst Search
---------------------------
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt

# Hack to fix import issue in older versions of pymc
import scipy
import scipy.misc
scipy.derivative = scipy.misc.derivative
import pymc

from astroML.decorators import pickle_results
from astroML.plotting.mcmc import plot_mcmc


#----------------------------------------------------------------------
# Set up toy dataset
def chirp(t, T, A, phi, omega, beta):
    """chirp signal"""
    signal = A * np.sin(phi + omega * (t - T) + beta * (t - T) ** 2)
    signal[t < T] = 0
    return signal


def background(t, b0, b1, Omega1, Omega2):
    """background signal"""
    return b0 + b1 * np.sin(Omega1 * t) * np.sin(Omega2 * t)


np.random.seed(0)
N = 500
T_true = 30
A_true = 0.8
phi_true = np.pi / 2
omega_true = 0.1
beta_true = 0.02
b0_true = 0.5
b1_true = 1.0
Omega1_true = 0.3
Omega2_true = 0.4
sigma = 0.1

t = 100 * np.random.random(N)

signal = chirp(t, T_true, A_true, phi_true, omega_true, beta_true)
bg = background(t, b0_true, b1_true, Omega1_true, Omega2_true)

y_true = signal + bg

y_obs = np.random.normal(y_true, sigma)

t_fit = np.linspace(0, 100, 1000)
y_fit = (chirp(t_fit, T_true, A_true, phi_true, omega_true, beta_true) +
         background(t_fit, b0_true, b1_true, Omega1_true, Omega2_true))


#----------------------------------------------------------------------
# Set up MCMC sampling
T = pymc.Uniform('T', 0, 100, value=T_true)
A = pymc.Uniform('A', 0, 100, value=A_true)
phi = pymc.Uniform('phi', -np.pi, np.pi, value=phi_true)
log_omega = pymc.Uniform('log_omega', -4, 0, value=np.log(omega_true))
log_beta = pymc.Uniform('log_beta', -6, 0, value=np.log(beta_true))
b0 = pymc.Uniform('b0', 0, 100, value=b0_true)
b1 = pymc.Uniform('b1', 0, 100, value=b1_true)
log_Omega1 = pymc.Uniform('log_Omega1', -3, 0, value=np.log(Omega1_true))
log_Omega2 = pymc.Uniform('log_Omega2', -3, 0, value=np.log(Omega2_true))
omega = pymc.Uniform('omega', 0.001, 1, value=omega_true)
beta = pymc.Uniform('beta', 0.001, 1, value=beta_true)


# uniform prior on log(Omega1)
@pymc.deterministic
def Omega1(log_Omega1=log_Omega1):
    return np.exp(log_Omega1)


# uniform prior on log(Omega2)
@pymc.deterministic
def Omega2(log_Omega2=log_Omega2):
    return np.exp(log_Omega2)


@pymc.deterministic
def y_model(t=t, T=T, A=A, phi=phi, omega=omega, beta=beta,
            b0=b0, b1=b1, Omega1=Omega1, Omega2=Omega2):
    return (chirp(t, T, A, phi, omega, beta)
            + background(t, b0, b1, Omega1, Omega2))

y = pymc.Normal('y', mu=y_model, tau=sigma ** -2, observed=True, value=y_obs)

model = dict(T=T, A=A, phi=phi, b0=b0, b1=b1,
             log_omega=log_omega, omega=omega,
             log_beta=log_beta, beta=beta,
             log_Omega1=log_Omega1, Omega1=Omega1,
             log_Omega2=log_Omega2, Omega2=Omega2,
             y_model=y_model, y=y)


#----------------------------------------------------------------------
# Run the MCMC sampling (saving the results to a pickle file)
@pickle_results('matchedfilt_chirp2.pkl')
def compute_MCMC(niter=30000, burn=2000):
    S = pymc.MCMC(model)
    S.sample(iter=30000, burn=2000)
    traces = [S.trace(s)[:] for s in ['T', 'A', 'omega', 'beta']]
    return traces

traces = compute_MCMC()

labels = ['$T$', '$A$', r'$\omega$', r'$\beta$']
limits = [(29.75, 30.25), (0.75, 0.83), (0.085, 0.115), (0.0197, 0.0202)]
true = [T_true, A_true, omega_true, beta_true]

#------------------------------------------------------------
# Plot results
fig = plt.figure(figsize=(8, 8))

# This function plots multiple panels with the traces
axes_list = plot_mcmc(traces, labels=labels, limits=limits,
                      true_values=true, fig=fig,
                      bins=30, colors='k', linewidths=2,
                      bounds=[0.14, 0.08, 0.95, 0.95])

for ax in axes_list:
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(plt.MaxNLocator(5))

ax = fig.add_axes([0.5, 0.7, 0.45, 0.25])
ax.scatter(t, y_obs, s=9, lw=0, c='k')
ax.plot(t_fit, y_fit, '-k')
ax.set_xlim(0, 100)
ax.set_xlabel('$t$')
ax.set_ylabel('$h_{obs}$')

plt.show()
