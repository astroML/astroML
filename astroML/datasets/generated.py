import numpy as np
from ..cosmology import Cosmology
from ..density_estimation import FunctionDistribution
from ..utils import check_random_state


def redshift_distribution(z, z0):
    return (z / z0) ** 2 * np.exp(-1.5 * (z / z0))


def generate_mu_z(size=1000, z0=0.3, dmu_0=0.1, dmu_1=0.02,
                  random_state=None, **kwargs):
    """Generate a dataset of distance modulus vs redshift.

    Parameters
    ----------
    size : int or tuple
        size of generated data
    z0 : float
        parameter in redshift distribution:
        p(z) ~ (z / z0)^2 exp[-1.5 (z / z0)]
    dmu_0, dmu_1 : float
        specify the error in mu, dmu = dmu_0 + dmu_1 * mu
    random_state : None, int, or np.random.RandomState instance
        random seed or random number generator
    **kwargs :
        additional keyword arguments are passed to the Cosmology function

    Returns
    -------
    z, mu, dmu : ndarrays
        arrays of shape `size`
    """
    random_state = check_random_state(random_state)
    cosmo = Cosmology(**kwargs)

    zdist = FunctionDistribution(redshift_distribution, func_args=dict(z0=z0),
                                 xmin=0.1 * z0, xmax=10 * z0,
                                 random_state=random_state)

    z_sample = zdist.rvs(size)
    mu_sample = np.reshape([cosmo.mu(z) for z in z_sample.ravel()], size)
    dmu = dmu_0 + dmu_1 * mu_sample
    mu_sample = random_state.normal(mu_sample, dmu)

    return z_sample, mu_sample, dmu
