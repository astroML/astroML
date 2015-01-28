import os
import numpy as np

from . import get_data_home
from .tools import download_with_progress_bar

DATA_URL = ('http://lambda.gsfc.nasa.gov/data/map/dr4/'
            'skymaps/7yr/raw/wmap_band_imap_r9_7yr_W_v4.fits')
MASK_URL = ('http://lambda.gsfc.nasa.gov/data/map/dr4/'
            'ancillary/masks/wmap_temperature_analysis_mask_r9_7yr_v4.fits')


def fetch_wmap_temperatures(masked=False, data_home=None,
                            download_if_missing=True):
    """Loader for WMAP temperature map data

    Parameters
    ----------
    masked : optional, default=False
        If True, then return the foreground-masked healpix array of data
        If False, then return the raw temperature array
    data_home : optional, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit learn data is stored in '~/astroML_data' subfolders.

    download_if_missing : optional, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    data : np.ndarray or np.ma.MaskedArray
        record array containing (masked) temperature data
    """
    # because of a bug in healpy, pylab must be imported before healpy is
    # or else a segmentation fault can result.
    import pylab
    import healpy as hp

    data_home = get_data_home(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    data_file = os.path.join(data_home, os.path.basename(DATA_URL))
    mask_file = os.path.join(data_home, os.path.basename(MASK_URL))

    if not os.path.exists(data_file):
        if not download_if_missing:
            raise IOError('data not present on disk. '
                          'set download_if_missing=True to download')
        data_buffer = download_with_progress_bar(DATA_URL)
        open(data_file, 'wb').write(data_buffer)

    data = hp.read_map(data_file)

    if masked:
        if not os.path.exists(mask_file):
            if not download_if_missing:
                raise IOError('mask data not present on disk. '
                              'set download_if_missing=True to download')
            mask_buffer = download_with_progress_bar(MASK_URL)
            open(mask_file, 'w').write(mask_buffer)

        mask = hp.read_map(mask_file)

        data = hp.ma(data)
        data.mask = np.logical_not(mask)  # WMAP mask has 0=bad. We need 1=bad

    return data
