import numpy as np
from scipy import integrate


class Cosmology(object):
    """Class to enable simple cosmological calculations.

    For a more full-featured cosmology package, see CosmoloPy [1]_

    Parameters
    ----------
    omegaM : float
        Matter Density. 0 <= omegaM <= 1
    omegaL : float
        Dark energy density. 0 <= omegaL <= 1
    h : float
        Hubble parameter, in units of 100 km/s/Mpc

    References
    ----------
    [1] http://roban.github.com/CosmoloPy/
    """
    def __init__(self, omegaM=0.27, omegaL=0.73, h=0.71):
        self.omegaM = omegaM
        self.omegaL = omegaL
        self.omegaK = 1. - omegaM - omegaL
        self.h = h

        # compute hubble distance in Mpc
        self.Dh = 2.9979E5 / (100 * h)

    def _hinv(self, z):
        """
        dimensionless Hubble constant at redshift z
        This is used in integration routines
        Defined as in equation 14 from Hogg 1999, and modified
        for non-constant w parameterized linearly with z ( w = w0 + w1*z )
        """
        if np.isinf(z):
            return np.inf
        return np.sqrt(self.omegaM * (1. + z) ** 3
                       + self.omegaK * (1. + z) ** 2
                       + self.omegaL)

    def Dc(self, z):
        """
        Line of sight comoving distance at redshift z
        Remains constant with epoch if objects are in the Hubble flow
        """
        if z == 0:
            return 0
        else:
            f = lambda z: 1.0 / self._hinv(z)
            I = integrate.quad(f, 0, z)
            return self.Dh * I[0]

    def Dm(self, z):
        """
        Transverse comoving distance at redshift z
        At same redshift but separated by angle dtheta;
        Dm * dtheta is transverse comoving distance
        """
        sOk = np.sqrt(abs(self.omegaK))

        if self.omegaK < 0.0:
            return self.Dh * np.sin(sOk * self.Dc(z) / self.Dh) / sOk
        elif self.omegaK == 0.0:
            return self.Dc(z)
        else:
            return self.Dh * np.sinh(sOk * self.Dc(z) / self.Dh) / sOk

    def Dl(self, z):
        """Luminosity distance (Mpc) at redshift z"""
        return (1. + z) * self.Dm(z)

    def mu(self, z):
        """Distance Modulus at redshift z"""
        return 5. * np.log10(self.Dl(z) * 1E6) - 5.
