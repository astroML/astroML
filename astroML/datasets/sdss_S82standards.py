import os
from gzip import GzipFile
from io import BytesIO

import numpy as np

from .tools import download_with_progress_bar
from . import get_data_home

DATA_URL = ('https://github.com/astroML/astroML-data/raw/master/datasets/'
            'stripe82calibStars_v2.6.dat.gz')
DATA_URL_2MASS = ('https://github.com/astroML/astroML-data/raw/master/datasets/'
                  'stripe82calibStars_2MASS_v2.6.dat.gz')

ARCHIVE_FILE = 'sdss_S82standards.npy'
ARCHIVE_FILE_2MASS = 'sdss_S82standards_2mass.npy'

DTYPE = [('RA', 'f8'),
         ('DEC', 'f8'),
         ('RArms', 'f4'),
         ('DECrms', 'f4'),
         ('Ntot', 'i4'),
         ('A_r', 'f4')]

for band in 'ugriz':
    DTYPE += [('Nobs_%s' % band, 'i4')]
    DTYPE += map(lambda s: (s + '_' + band, 'f4'),
                 ['mmed', 'mmu', 'msig', 'mrms', 'mchi2'])

DTYPE_2MASS = DTYPE + [('ra2MASS', 'f4'),
                       ('dec2MASS', 'f4'),
                       ('J', 'f4'),
                       ('Jerr', 'f4'),
                       ('H', 'f4'),
                       ('Herr', 'f4'),
                       ('K', 'f4'),
                       ('Kerr', 'f4'),
                       ('theta', 'f4')]

# first column is 'CALIBSTARS'.  We'll ignore this.
COLUMNS = range(1, len(DTYPE) + 1)


def fetch_sdss_S82standards(data_home=None, download_if_missing=True,
                            crossmatch_2mass=False):
    """Loader for SDSS stripe82 standard star catalog

    Parameters
    ----------
    data_home : optional, default=None
        Specify another download and cache folder for the datasets. By default
        all astroML data is stored in '~/astroML_data'.

    download_if_missing : bool, optional, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    crossmatch_2mass: bool, optional, default=False
        If True, return the standard star catalog cross-matched with 2mass
        magnitudes

    Returns
    -------
    data : ndarray, shape = (313859,)
        record array containing sdss standard stars (see notes below)

    Notes
    -----
    Information on the data can be found at
    http://www.astro.washington.edu/users/ivezic/sdss/catalogs/stripe82.html
    Data is described in Ivezic et al. 2007 (Astronomical Journal, 134, 973).
    Columns are as follows:

       RA                Right-ascention of source (degrees)
       DEC               Declination of source (degrees)
       RArms             rms of right-ascention (arcsec)
       DECrms            rms of declination (arcsec)
       Ntot              total number of epochs
       A_r               SFD ISM extinction (mags)

       for each band in (u g r i z):
           Nobs_<band>    number of observations in this band
           mmed_<band>    median magnitude in this band
           mmu_<band>     mean magnitude in this band
           msig_<band>    standard error on the mean
                          (1.25 times larger for median)
           mrms_<band>    root-mean-square scatter
           mchi2_<band>   chi2 per degree of freedom for mean magnitude

    For 2-MASS, the following columns are added:

       ra2MASS           2-mass right-ascention
       dec2MASS          2-mass declination
       J                 J-band magnitude
       Jerr              J-band error
       H                 H-band magnitude
       Herr              H-band error
       K                 K-band magnitude
       Kerr              K-band error
       theta             difference between SDSS and 2MASS position (arcsec)

    Examples
    --------
    >>> data = fetch_sdss_S82standards()  # doctest: +IGNORE_OUTPUT +REMOTE_DATA
    >>> u_g = data['mmed_u'] - data['mmed_g']  # doctest: +REMOTE_DATA
    >>> print(u_g[:4])  # doctest: +REMOTE_DATA
    [-22.23500061   1.34900093   1.43799973   2.08200073]

    References
    ----------
    Ivesic et al. ApJ 134:973 (2007)
    """
    data_home = get_data_home(data_home)

    if crossmatch_2mass:
        archive_file = os.path.join(data_home, ARCHIVE_FILE_2MASS)
        data_url = DATA_URL_2MASS
        kwargs = dict(dtype=DTYPE_2MASS)

    else:
        archive_file = os.path.join(data_home, ARCHIVE_FILE)
        data_url = DATA_URL
        kwargs = dict(usecols=COLUMNS, dtype=DTYPE)

    if not os.path.exists(archive_file):
        if not download_if_missing:
            raise IOError('data not present on disk. '
                          'set download_if_missing=True to download')

        print("downloading cross-matched SDSS/2MASS dataset from %s to %s"
              % (data_url, data_home))

        zipped_buf = download_with_progress_bar(data_url, return_buffer=True)
        gzf = GzipFile(fileobj=zipped_buf, mode='rb')
        print("uncompressing file...")
        extracted_buf = BytesIO(gzf.read())
        data = np.loadtxt(extracted_buf, **kwargs)
        np.save(archive_file, data)

    else:
        data = np.load(archive_file)

    return data
