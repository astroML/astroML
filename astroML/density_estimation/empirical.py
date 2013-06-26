import numpy as np
from scipy import interpolate
from ..utils import check_random_state


class FunctionDistribution(object):
    """Generate random variables distributed according to an arbitrary function

    Parameters
    ----------
    func : function
        func should take an array of x values, and return an array
        proportional to the probability density at each value
    xmin : float
        minimum value of interest
    xmax : float
        maximum value of interest
    Nx : int (optional)
        number of samples to draw.  Default is 1000
    random_state : None, int, or np.random.RandomState instance
        random seed or random number generator
    func_args : dictionary (optional)
        additional keyword arguments to be passed to func
    """
    def __init__(self, func, xmin, xmax, Nx=1000,
                 random_state=None, func_args=None):
        self.random_state = check_random_state(random_state)

        if func_args is None:
            func_args = {}

        x = np.linspace(xmin, xmax, Nx)
        Px = func(x, **func_args)

        # if there are too many zeros, interpolation will fail
        positive = (Px > 1E-10 * Px.max())
        x = x[positive]
        Px = Px[positive].cumsum()
        Px /= Px[-1]

        self._tck = interpolate.splrep(Px, x)

    def rvs(self, shape):
        """Draw random variables from the distribution

        Parameters
        ----------
        shape : integer or tuple
            shape of desired array

        Returns
        -------
        rv : ndarray, shape=shape
            random variables
        """
        # generate uniform variables between 0 and 1
        y = self.random_state.random_sample(shape)
        return interpolate.splev(y, self._tck)


class EmpiricalDistribution(object):
    """Empirically learn a distribution from one-dimensional data

    Parameters
    ----------
    data : one-dimensional array
        input data

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> x = np.random.normal(size=10000)  # normally-distributed variables
    >>> x.mean(), x.std()
    (-0.018433720158265783, 0.98755656817612003)
    >>> x2 = EmpiricalDistribution(x).rvs(10000)
    >>> x2.mean(), x2.std()
    (-0.020293716681613363, 1.0039249294845276)

    Notes
    -----
    This function works by approximating the inverse of the cumulative
    distribution using an efficient spline fit to the sorted values.
    """
    def __init__(self, data):
        # copy, because we'll need to sort in-place
        data = np.array(data, copy=True)
        if data.ndim != 1:
            raise ValueError("data should be one-dimensional")
        data.sort()

        # set up spline
        y = np.linspace(0, 1, data.size)
        self._tck = interpolate.splrep(y, data)

    def rvs(self, shape):
        """Draw random variables from the distribution

        Parameters
        ----------
        shape : integer or tuple
            shape of desired array

        Returns
        -------
        rv : ndarray, shape=shape
            random variables
        """
        # generate uniform variables between 0 and 1
        y = np.random.random(shape)
        return interpolate.splev(y, self._tck)
