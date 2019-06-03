import numpy as np
from sklearn.utils import check_random_state


def generate_power_law(N, dt, beta, generate_complex=False, random_state=None):
    """Generate a power-law light curve

    This uses the method from Timmer & Koenig [1]_

    Parameters
    ----------
    N : integer
        Number of equal-spaced time steps to generate
    dt : float
        Spacing between time-steps
    beta : float
        Power-law index.  The spectrum will be (1 / f)^beta
    generate_complex : boolean (optional)
        if True, generate a complex time series rather than a real time series
    random_state : None, int, or np.random.RandomState instance (optional)
        random seed or random number generator

    Returns
    -------
    x : ndarray
        the length-N

    References
    ----------
    .. [1] Timmer, J. & Koenig, M. On Generating Power Law Noise. A&A 300:707
    """
    random_state = check_random_state(random_state)
    dt = float(dt)
    N = int(N)

    Npos = int(N / 2)
    Nneg = int((N - 1) / 2)
    domega = (2 * np.pi / dt / N)

    if generate_complex:
        omega = domega * np.fft.ifftshift(np.arange(N) - int(N / 2))
    else:
        omega = domega * np.arange(Npos + 1)

    x_fft = np.zeros(len(omega), dtype=complex)
    x_fft.real[1:] = random_state.normal(0, 1, len(omega) - 1)
    x_fft.imag[1:] = random_state.normal(0, 1, len(omega) - 1)

    x_fft[1:] *= (1. / omega[1:]) ** (0.5 * beta)
    x_fft[1:] *= (1. / np.sqrt(2))

    # by symmetry, the Nyquist frequency is real if x is real
    if (not generate_complex) and (N % 2 == 0):
        x_fft.imag[-1] = 0

    if generate_complex:
        x = np.fft.ifft(x_fft)
    else:
        x = np.fft.irfft(x_fft, N)

    return x


def generate_damped_RW(t_rest, tau=300., z=2.0,
                       xmean=0, SFinf=0.3, random_state=None):
    """Generate a damped random walk light curve

    This uses a damped random walk model to generate a light curve similar
    to that of a QSO [1]_.

    Parameters
    ----------
    t_rest : array_like
        rest-frame time.  Should be in increasing order
    tau : float
        relaxation time
    z : float
        redshift
    xmean : float (optional)
        mean value of random walk; default=0
    SFinf : float (optional
        Structure function at infinity; default=0.3
    random_state : None, int, or np.random.RandomState instance (optional)
        random seed or random number generator

    Returns
    -------
    x : ndarray
        the sampled values corresponding to times t_rest

    Notes
    -----
    The differential equation is (with t = time/tau):

        dX = -X(t) * dt + sigma * sqrt(tau) * e(t) * sqrt(dt) + b * tau * dt

    where e(t) is white noise with zero mean and unit variance, and

        Xmean = b * tau
        SFinf = sigma * sqrt(tau / 2)

    so

        dX(t) = -X(t) * dt + sqrt(2) * SFint * e(t) * sqrt(dt) + Xmean * dt

    References
    ----------
    .. [1] Kelly, B., Bechtold, J. & Siemiginowska, A. (2009)
           Are the Variations in Quasar Optical Flux Driven by Thermal
           Fluctuations? ApJ 698:895 (2009)
    """
    #  Xmean = b * tau
    #  SFinf = sigma * sqrt(tau / 2)
    t_rest = np.atleast_1d(t_rest)

    if t_rest.ndim != 1:
        raise ValueError('t_rest should be a 1D array')

    random_state = check_random_state(random_state)

    N = len(t_rest)

    t_obs = t_rest * (1. + z) / tau

    x = np.zeros(N)
    x[0] = random_state.normal(xmean, SFinf)
    E = random_state.normal(0, 1, N)

    for i in range(1, N):
        dt = t_obs[i] - t_obs[i - 1]
        x[i] = (x[i - 1]
                - dt * (x[i - 1] - xmean)
                + np.sqrt(2) * SFinf * E[i] * np.sqrt(dt))

    return x
