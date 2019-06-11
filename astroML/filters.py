import numpy as np
from scipy import optimize, fftpack, signal

from astroML.utils.decorators import deprecated
from astroML.utils.exceptions import AstroMLDeprecationWarning

# Note: there is a scipy PR to include an improved SG filter within the
# scipy.signal submodule.  It should replace this when it's finished.
# see http://github.com/scipy/scipy/pull/304
@deprecated('1.0', alternative='scipy.signal.savgol_filter',
            warning_type=AstroMLDeprecationWarning)
def savitzky_golay(y, window_size, order, deriv=0,
                   use_fft=True):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter

    This implementation is based on [1]_.

    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techhniques.

    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute
        (default = 0 means only smoothing)
    use_fft : bool
        if True (default) then convolue using FFT for speed

    Returns
    -------
    y_smooth : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).

    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    Examples
    --------
    >>> t = np.linspace(-4, 4, 500)
    >>> y = np.exp(-t ** 2)
    >>> y_smooth = savitzky_golay(y, window_size=31, order=4)

    References
    ----------
    .. [1] http://www.scipy.org/Cookbook/SavitzkyGolay
    .. [2] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [3] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")

    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order + 1)

    half_window = (window_size - 1) // 2

    # precompute coefficients
    b = np.array([[k ** i for i in order_range]
                  for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b)[deriv]

    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])

    y = np.concatenate((firstvals, y, lastvals))

    if use_fft:
        return signal.fftconvolve(y, m, mode='valid')
    else:
        return np.convolve(y, m, mode='valid')


def wiener_filter(t, h, signal='gaussian', noise='flat', return_PSDs=False,
                  signal_params=None, noise_params=None):
    """Compute a Wiener-filtered time-series

    Parameters
    ----------
    t : array_like
        evenly-sampled time series, length N
    h : array_like
        observations at each t
    signal : str (optional)
        currently only 'gaussian' is supported
    noise : str (optional)
        currently only 'flat' is supported
    return_PSDs : bool (optional)
        if True, then return (PSD, P_S, P_N)
    signal_guess : tuple (optional)
        A starting guess at the parameters for the signal.  If not specified,
        a suitable guess will be estimated from the data itself. (see Notes
        below)
    noise_guess : tuple (optional)
        A starting guess at the parameters for the noise.  If not specified,
        a suitable guess will be estimated from the data itself. (see Notes
        below)

    Returns
    -------
    h_smooth : ndarray
        a smoothed version of h, length N

    Notes
    -----
    The Wiener filter operates by fitting a functional form to the PSD::

       PSD = P_S + P_N

    The resulting frequency-space filter is given by::

       Phi = P_S / (P_S + P_N)

    This entire operation is equivalent to a kernel smoothing by a
    kernel whose Fourier transform is Phi.

    Choosing Signal/Noise Parameters
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    the arguments ``signal_guess`` and ``noise_guess`` specify the initial
    guess for the characteristics of signal and noise used in the minimization.
    They are generally expected to be tuples, and the meaning varies depending
    on the form of signal and noise used.  For ``gaussian``, the params are
    (amplitude, width).  For ``flat``, the params are (amplitude,).

    See Also
    --------
    scipy.signal.wiener : a static (non-adaptive) wiener filter
    """
    # Validate signal
    if signal != 'gaussian':
        raise ValueError("only signal='gaussian' is supported")
    if signal_params is not None and len(signal_params) != 2:
        raise ValueError("signal_params should be length 2")

    # Validate noise
    if noise != 'flat':
        raise ValueError("only noise='flat' is supported")
    if noise_params is not None and len(noise_params) != 1:
        raise ValueError("noise_params should be length 1")

    # Validate t and hd
    t = np.asarray(t)
    h = np.asarray(h)

    if (t.ndim != 1) or (t.shape != h.shape):
        raise ValueError('t and h must be equal-length 1-dimensional arrays')

    # compute the PSD of the input
    N = len(t)
    Df = 1. / N / (t[1] - t[0])
    f = fftpack.ifftshift(Df * (np.arange(N) - N / 2))

    H = fftpack.fft(h)
    PSD = abs(H) ** 2

    # fit signal/noise params if necessary
    if signal_params is None:
        amp_guess = np.max(PSD[1:])
        width_guess = np.min(np.abs(f[PSD < np.mean(PSD[1:])]))
        signal_params = (amp_guess, width_guess)
    if noise_params is None:
        noise_params = (np.mean(PSD[1:]),)

    # Set up the Wiener filter:
    #  fit a model to the PSD: sum of signal form and noise form

    def signal(x, A, width):
        width = abs(width) + 1E-99  # prevent divide-by-zero errors
        return A * np.exp(-0.5 * (x / width) ** 2)

    def noise(x, n):
        return n * np.ones(x.shape)

    # use [1:] here to remove the zero-frequency term: we don't want to
    # fit to this for data with an offset.
    min_func = lambda v: np.sum((PSD[1:] - signal(f[1:], v[0], v[1])
                                 - noise(f[1:], v[2])) ** 2)
    v0 = tuple(signal_params) + tuple(noise_params)
    v = optimize.minimize(min_func, v0, method='Nelder-Mead')['x']

    P_S = signal(f, v[0], v[1])
    P_N = noise(f, v[2])
    Phi = P_S / (P_S + P_N)
    Phi[0] = 1  # correct for DC offset

    # Use Phi to filter and smooth the values
    h_smooth = fftpack.ifft(Phi * H)

    if not np.iscomplexobj(h):
        h_smooth = h_smooth.real

    if return_PSDs:
        return h_smooth, PSD, P_S, P_N, Phi
    else:
        return h_smooth


def min_component_filter(x, y, feature_mask, p=1, fcut=None, Q=None):
    """Minimum component filtering

    Minimum component filtering is useful for determining the background
    component of a signal in the presence of spikes

    Parameters
    ----------
    x : array_like
        1D array of evenly spaced x values
    y : array_like
        1D array of y values corresponding to x
    feature_mask : array_like
        1D mask array giving the locations of features in the data which
        should be ignored for smoothing
    p : integer (optional)
        polynomial degree to be used for the fit (default = 1)
    fcut : float (optional)
        the cutoff frequency for the low-pass filter.  Default value is
        f_nyq / sqrt(N)
    Q : float (optional)
        the strength of the low-pass filter.  Larger Q means a steeper cutoff
        default value is 0.1 * fcut

    Returns
    -------
    y_filtered : ndarray
        The filtered version of y.

    Notes
    -----
    This code follows the procedure explained in the book
    "Practical Statistics for Astronomers" by Wall & Jenkins book, as
    well as in Wall, J, A&A 122:371, 1997
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    feature_mask = np.asarray(feature_mask, dtype=bool)

    if ((x.ndim != 1) or (x.shape != y.shape) or (y.shape !=
                                                  feature_mask.shape)):
        raise ValueError('x, y, and feature_mask must be 1 dimensional '
                         'with matching lengths')

    if fcut is None:
        f_nyquist = 1. / (x[1] - x[0])
        fcut = f_nyquist / np.sqrt(len(x))

    if Q is None:
        Q = 0.1 * fcut

    # compute polynomial features
    XX = x[:, None] ** np.arange(p + 1)

    # compute least-squares fit to non-masked data
    beta = np.linalg.lstsq(XX[~feature_mask], y[~feature_mask], rcond=None)[0]

    # subtract polynomial fit and mask the data
    y_mask = y - np.dot(XX, beta)
    y_mask[feature_mask] = 0

    # get Fourier transforms of arrays
    yFT_mask = fftpack.fft(y_mask)

    # compute (shifted) frequency array for filter
    N = len(x)
    f = fftpack.ifftshift((np.arange(N) - N / 2.) * 1. / N / (x[1] - x[0]))

    # construct low-pass filter
    filt = np.exp(- (Q * (abs(f) - fcut) / fcut) ** 2)
    filt[abs(f) < fcut] = 1

    # reconstruct filtered signal
    y_filtered = fftpack.ifft(yFT_mask * filt).real + np.dot(XX, beta)

    return y_filtered
