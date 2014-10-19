from __future__ import print_function, division

import os

import numpy as np

from . import get_data_home
from . import fetch_sdss_S82standards
from .tools import download_with_progress_bar

DATA_URL = ("http://www.astro.washington.edu/users/"
            "ivezic/DMbook/data/RRLyrae.fit")


def fetch_rrlyrae_mags(data_home=None, download_if_missing=True):
    """Loader for RR-Lyrae data

    Parameters
    ----------
    data_home : optional, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit learn data is stored in '~/astroML_data' subfolders.

    download_if_missing : optional, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    data : recarray, shape = (483,)
        record array containing imaging data

    Examples
    --------
    >>> from astroML.datasets import fetch_rrlyrae_mags
    >>> data = fetch_rrlyrae_mags()
    >>> data.shape  # number of objects in dataset
    (483,)
    >>> print(data.names[:5])  # names of the first five columns
    ['ra', 'dec', 'run', 'rExtSFD', 'uRaw']
    >>> print(data['ra'][:2])
    [ 0.265165  0.265413]
    >>> print(data['dec'][:2])
    [-0.444861 -0.62201 ]

    Notes
    -----
    This data is from table 1 of Sesar et al 2010 ApJ 708:717
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
    return np.asarray(hdulist[1].data)


def fetch_rrlyrae_combined(data_home=None, download_if_missing=True):
    """Loader for RR-Lyrae combined data

    This returns the combined RR-Lyrae colors and SDSS standards colors.
    The RR-Lyrae sample is confirmed through time-domain observations;
    this result in a nice dataset for testing classification routines.

    Parameters
    ----------
    data_home : optional, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit learn data is stored in '~/astroML_data' subfolders.

    download_if_missing : optional, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    X : ndarray
        a shape (n_samples, 4) array.  Columns are u-g, g-r, r-i, i-z

    y : ndarray
        a shape (n_samples,) array of labels.  1 indicates an RR Lyrae,
        0 indicates a background star.
    """
    #----------------------------------------------------------------------
    # Load data
    kwds = dict(data_home=data_home,
                download_if_missing=download_if_missing)
    rrlyrae = fetch_rrlyrae_mags(**kwds)
    standards = fetch_sdss_S82standards(**kwds)

    #------------------------------------------------------------
    # perform color cuts on standard stars
    # these come from eqns 1-4 of Sesar et al 2010, ApJ 708:717

    u_g = standards['mmu_u'] - standards['mmu_g']
    g_r = standards['mmu_g'] - standards['mmu_r']
    r_i = standards['mmu_r'] - standards['mmu_i']
    i_z = standards['mmu_i'] - standards['mmu_z']

    standards = standards[(u_g > 0.7) & (u_g < 1.35) &
                          (g_r > -0.15) & (g_r < 0.4) &
                          (r_i > -0.15) & (r_i < 0.22) &
                          (i_z > -0.21) & (i_z < 0.25)]

    #----------------------------------------------------------------------
    # get magnitudes and colors; split into train and test sets

    mags_rr = np.vstack([rrlyrae[f + 'mag'] for f in 'ugriz'])
    colors_rr = mags_rr[:-1] - mags_rr[1:]

    mags_st = np.vstack([standards['mmu_' + f] for f in 'ugriz'])
    colors_st = mags_st[:-1] - mags_st[1:]

    # stack the two sets of colors together
    X = np.vstack((colors_st.T, colors_rr.T))
    y = np.zeros(X.shape[0])
    y[-colors_rr.shape[1]:] = 1

    return X, y
