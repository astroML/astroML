import numpy as np
from astropy.cosmology import FlatLambdaCDM

from ..density_estimation import FunctionDistribution
from sklearn.utils import check_random_state


def redshift_distribution(z, z0):
    return (z / z0) ** 2 * np.exp(-1.5 * (z / z0))


def generate_mu_z(size=1000, z0=0.3, dmu_0=0.1, dmu_1=0.02,
                  random_state=None, cosmo=None):
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

    cosmo : astropy.cosmology instance specifying cosmology
        to use when generating the sample.  If not provided,
        a Flat Lambda CDM model with H0=71, Om0=0.27, Tcmb=0 is used.

    Returns
    -------
    z, mu, dmu : ndarrays
        arrays of shape ``size``
    """

    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=71, Om0=0.27, Tcmb0=0)

    random_state = check_random_state(random_state)
    zdist = FunctionDistribution(redshift_distribution, func_args=dict(z0=z0),
                                 xmin=0.1 * z0, xmax=10 * z0,
                                 random_state=random_state)

    z_sample = zdist.rvs(size)
    mu_sample = cosmo.distmod(z_sample).value

    dmu = dmu_0 + dmu_1 * mu_sample
    mu_sample = random_state.normal(mu_sample, dmu)

    return z_sample, mu_sample, dmu
