import os
import sys
import urllib2

import numpy as np

from .tools import get_data_home

SPEC_FILENAME = 'bc2003.ssp.m0.age18.tau0.npz'

SPEC_TOPLEVEL_URL = 'http://www.astro.washington.edu/users/ajc/teaching/a597/'
SPEC_URL = os.path.join(SPEC_TOPLEVEL_URL,
                        'homework/homework2/bc2003.ssp.m0.age18.tau0')

def fetch_spectrum(data_home=None, download_if_missing=True):
    """Loader for single spectrum for sum-of-norms fitting

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all astroML data is stored in '~/astroML_data' subfolders.

    download_if_missing: boolean, optional, default: True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    lam, flux
        ndarrays giving the wavelength in angstroms, and the associated flux
        of the spectrum
    """
    data_home = get_data_home(data_home)

    target_file = os.path.join(data_home, SPEC_FILENAME)

    if not os.path.exists(target_file):
        if not download_if_missing:
            raise IOError("Spectrum data not found")
        print ("downloading sdss colors training data from %s to %s" 
               % (SPEC_URL, target_file))

        # create a password manager
        password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()

        # Add the username and password.
        # If we knew the realm, we could use it instead of None.
        password_mgr.add_password(None, SPEC_TOPLEVEL_URL,
                                  'teaching', 'TeachMe')
        handler = urllib2.HTTPBasicAuthHandler(password_mgr)

        # create "opener" (OpenerDirector instance)
        opener = urllib2.build_opener(handler)

        # use the opener to fetch a URL
        fhandle = opener.open(SPEC_URL)

        lam, flux = np.loadtxt(fhandle).T
        flux *= 1E4
        np.savez(target_file, lam=lam, flux=flux)

    else:
        file_obj = np.load(target_file)
        lam = file_obj['lam']
        flux = file_obj['flux']

    return lam, flux
