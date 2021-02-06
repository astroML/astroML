import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from astroML.linear_model import TLS_logL, LinearRegression


# TLS:
def get_m_b(beta):
    b = np.dot(beta, beta) / beta[1]
    m = -beta[0] / beta[1]
    return m, b


def plot_regressions(ksi, eta, x, y, sigma_x, sigma_y, add_regression_lines=False,
                     alpha_in=1, beta_in=0.5, basis='linear'):

    figure = plt.figure(figsize=(8, 6))
    ax = figure.add_subplot(111)
    ax.scatter(x, y, alpha=0.5)
    ax.errorbar(x, y, xerr=sigma_x, yerr=sigma_y, alpha=0.3, ls='')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    x0 = np.linspace(np.min(x) - 0.5, np.max(x) + 0.5, 20)

    # True regression line

    if alpha_in is not None and beta_in is not None:
        if basis == 'linear':
            y0 = alpha_in + x0 * beta_in
        elif basis == 'poly':
            y0 = alpha_in + beta_in[0] * x0 + beta_in[1] * x0 * x0 + beta_in[2] * x0 * x0 * x0

        ax.plot(x0, y0, color='black', label='True regression')
    else:
        y0 = None

    if add_regression_lines:
        for label, data, *target in [['fit no errors', x, y, 1],
                                     ['fit y errors only', x, y, sigma_y],
                                     ['fit x errors only', y, x, sigma_x]]:
            linreg = LinearRegression()
            linreg.fit(data[:, None], *target)
            if label == 'fit x errors only' and y0 is not None:
                x_fit = linreg.predict(y0[:, None])
                ax.plot(x_fit, y0, label=label)
            else:
                y_fit = linreg.predict(x0[:, None])
                ax.plot(x0, y_fit, label=label)

        # TLS
        X = np.vstack((x, y)).T
        dX = np.zeros((len(x), 2, 2))
        dX[:, 0, 0] = sigma_x
        dX[:, 1, 1] = sigma_y

        def min_func(beta): return -TLS_logL(beta, X, dX)
        beta_fit = optimize.fmin(min_func, x0=[-1, 1])
        m_fit, b_fit = get_m_b(beta_fit)
        x_fit = np.linspace(-10, 10, 20)
        ax.plot(x_fit, m_fit * x_fit + b_fit, label='TLS')

    ax.set_xlim(np.min(x)-0.5, np.max(x)+0.5)
    ax.set_ylim(np.min(y)-0.5, np.max(y)+0.5)
    ax.legend()


def plot_regression_from_trace(fitted, observed, ax=None, chains=None, multidim_ind=None):

    traces = [fitted.trace, ]
    xi, yi, sigx, sigy = observed

    if multidim_ind is not None:
        xi = xi[multidim_ind]

    x = np.linspace(np.min(xi)-0.5, np.max(xi)+0.5, 50)

    for i, trace in enumerate(traces):
        if 'theta' in trace.varnames and 'slope' not in trace.varnames:
            trace.add_values({'slope': np.tan(trace['theta'])})

        if multidim_ind is not None:
            trace_slope = trace['slope'][:, multidim_ind]
        else:
            trace_slope = trace['slope'][:, 0]

        if chains is not None:
            for chain in range(100, len(trace) * trace.nchains, chains):
                y = trace['inter'][chain] + trace_slope[chain] * x
                ax.plot(x, y, alpha=0.03, c='red')

        # plot the best-fit line only
        H2D, bins1, bins2 = np.histogram2d(trace_slope,
                                           trace['inter'], bins=50)

        w = np.where(H2D == H2D.max())

        # choose the maximum posterior slope and intercept
        slope_best = bins1[w[0][0]]
        intercept_best = bins2[w[1][0]]

        print("beta:", slope_best, "alpha:", intercept_best)
        y = intercept_best + slope_best * x

        # y_pre = fitted.predict(x[:, None])
        ax.plot(x, y, ':', label='fitted')

        ax.legend()
        break
