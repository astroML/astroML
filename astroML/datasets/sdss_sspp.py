from __future__ import print_function

import os

import numpy as np

from . import get_data_home
from .tools import download_with_progress_bar

DATA_URL = ("http://www.astro.washington.edu/users/ivezic/"
            "DMbook/data/SDSSssppDR9_rerun122.fit")


def compute_distances(data):
    """Compute the distances to select stars in the sdss_sspp sample.

    Distance are determined using empirical color/magnitude fits from
    Ivezic et al 2008, ApJ 684:287

    Extinction correcctions come from Berry et al 2011, arXiv 1111.4985

    This distance only works for stars with log(g) > 3.3
    Other stars will have distance=-1
    """
    # extinction terms from Berry et al
    Ar = data['Ar']
    Au = 1.810 * Ar
    Ag = 1.400 * Ar
    Ai = 0.759 * Ar
    Az = 0.561 * Ar

    # compute corrected mags and colors
    gmag = data['gpsf'] - Ag
    rmag = data['rpsf'] - Ar
    imag = data['ipsf'] - Ai
    gi = gmag - imag

    # compute distance fit from Ivezic et al
    FeH = data['FeH']
    Mr0 = (-5.06 + 14.32 * gi - 12.97 * gi ** 2 +
           6.127 * gi ** 3 - 1.267 * gi ** 4 + 0.0967 * gi ** 5)
    FeHoffset = 4.50 - 1.11 * FeH - 0.18 * FeH ** 2
    Mr = Mr0 + FeHoffset
    dist = 0.01 * 10 ** (0.2 * (rmag - Mr))

    # stars with log(g) < 3.3 don't work for this fit: set distance to -1
    dist[data['logg'] < 3.3] = -1

    return dist


def fetch_sdss_sspp(data_home=None, download_if_missing=True, cleaned=False):
    """Loader for SDSS SEGUE Stellar Parameter Pipeline data

    Parameters
    ----------
    data_home : optional, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit learn data is stored in '~/astroML_data' subfolders.

    download_if_missing : bool (optional) default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    cleaned : bool (optional) default=False
        if True, then return a cleaned catalog where objects with extreme
        values are removed.

    Returns
    -------
    data : recarray, shape = (327260,)
        record array containing pipeline parameters

    Notes
    -----
    Here are the comments from the fits file header:

    Imaging data and spectrum identifiers for a sample of 327,260
    stars with SDSS spectra,  selected as:

      1) available SSPP parameters in SDSS Data Release 9
         (SSPP rerun 122, file from Y.S. Lee)
      2) 14 < r < 21 (psf magnitudes, uncorrected for ISM extinction)
      3) 10 < u < 25 & 10 < z < 25 (same as above)
      4) errors in ugriz well measured (>0) and <10
      5) 0 < u-g < 3 (all color cuts based on psf mags, dereddened)
      6) -0.5 < g-r < 1.5 & -0.5 < r-i < 1.0 & -0.5 < i-z < 1.0
      7) -200 < pmL < 200 & -200 < pmB < 200 (proper motion in mas/yr)
      8) pmErr < 10 mas/yr (proper motion error)
      9) 1 < log(g) < 5
      10) TeffErr < 300 K

    Teff and TeffErr are given in Kelvin, radVel and radVelErr in km/s.
    (ZI, Feb 2012, ivezic@astro.washington.edu)

    Examples
    --------
    >>> from astroML.datasets import fetch_sdss_sspp
    >>> data = fetch_sdss_sspp()
    >>> data.shape  # number of objects in dataset
    (327260,)
    >>> print(data.names[:5])  # names of the first five columns
    ['ra', 'dec', 'Ar', 'upsf', 'uErr']
    >>> print(data['ra'][:2])  # first two RA values
    [ 49.62750244  40.27209091]
    >>> print(data['dec'][:2])  # first two DEC values
    [-1.04175591 -0.64250112]
    """
    # fits is an optional dependency: don't import globally
    from astropy.io import fits

    data_home = get_data_home(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    archive_file = os.path.join(data_home, os.path.basename(DATA_URL))

    if not os.path.exists(archive_file):
        if not download_if_missing:
            raise IOError('data not present on disk. '
                          'set download_if_missing=True to download')

        fitsdata = download_with_progress_bar(DATA_URL)
        open(archive_file, 'wb').write(fitsdata)

    hdulist = fits.open(archive_file)

    data = np.asarray(hdulist[1].data)

    if cleaned:
        # -1.1 < FeH < 0.1
        data = data[(data['FeH'] > -1.1) & (data['FeH'] < 0.1)]

        # -0.03 < alpha/Fe < 0.57
        data = data[(data['alphFe'] > -0.03) & (data['alphFe'] < 0.57)]

        # 5000 < Teff < 6500
        data = data[(data['Teff'] > 5000) & (data['Teff'] < 6500)]

        # 3.5 < log(g) < 5
        data = data[(data['logg'] > 3.5) & (data['logg'] < 5)]

        # 0 < error for FeH < 0.1
        data = data[(data['FeHErr'] > 0) & (data['FeHErr'] < 0.1)]

        # 0 < error for alpha/Fe < 0.05
        data = data[(data['alphFeErr'] > 0) & (data['alphFeErr'] < 0.05)]

        # 15 < g mag < 18
        data = data[(data['gpsf'] > 15) & (data['gpsf'] < 18)]

        # abs(radVel) < 100 km/s
        data = data[(abs(data['radVel']) < 100)]

    return data
