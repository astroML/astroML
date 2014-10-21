"""
SDSS DR7 Quasar Dataset Loader.

This implements a loader for the DR7 quasar dataset, located at
http://www.sdss.org/dr7/products/value added/qsocat_dr7.html
"""
from __future__ import print_function, division

import os
from gzip import GzipFile

import numpy as np

from .tools import download_with_progress_bar
from ..py3k_compat import BytesIO
from . import get_data_home

DATA_URL = 'http://das.sdss.org/va/qsocat/dr7qso.dat.gz'

ARCHIVE_FILE = 'dr7_quasar.npy'

#column numbers for extraction
DR7_DTYPE = [('sdssID', 'a14'),
             ('RA', 'f8'),
             ('dec', 'f8'),
             ('redshift', 'f4'),
             ('mag_u', 'f4'),
             ('err_u', 'f4'),
             ('mag_g', 'f4'),
             ('err_g', 'f4'),
             ('mag_r', 'f4'),
             ('err_r', 'f4'),
             ('mag_i', 'f4'),
             ('err_i', 'f4'),
             ('mag_z', 'f4'),
             ('err_z', 'f4'),
             ('mag_J', 'f4'),
             ('err_J', 'f4'),
             ('mag_H', 'f4'),
             ('err_H', 'f4'),
             ('mag_K', 'f4'),
             ('err_K', 'f4'),
             ('specobjid', 'i8')]

COLUMN_NUMBERS = [0, 1, 2, 3,
                  4, 5, 6, 7,
                  8, 9, 10, 11, 12, 13,
                  22, 23, 24, 25, 26, 27, 72]

# length of header information
SKIP_ROWS = 80


def fetch_dr7_quasar(data_home=None, download_if_missing=True):
    """Loader for SDSS DR7 quasar catalog

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
    data : ndarray, shape = (105783,)
        numpy record array containing the quasar catalog

    Examples
    --------
    >>> from astroML.datasets import fetch_dr7_quasar
    >>> data = fetch_dr7_quasar()
    >>> u_g = data['mag_u'] - data['mag_g']
    >>> u_g[:3]  # first three u-g colors
    array([-0.07699966,  0.03600121,  0.10900116], dtype=float32)

    Notes
    -----
    Not all available data is extracted and saved.  The extracted columns are:

    sdssID, RA, DEC, redshift, mag_u, err_u, mag_g, err_g, mag_r, err_r,
    mag_i, err_i, mag_z, err_z, mag_J, err_J, mag_H, err_H, mag_K, err_K,
    specobjid

    many of the objects are missing 2mass photometry.

    More information at
    http://www.sdss.org/dr7/products/value_added/qsocat_dr7.html
    """
    data_home = get_data_home(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    archive_file = os.path.join(data_home, ARCHIVE_FILE)

    if not os.path.exists(archive_file):
        if not download_if_missing:
            raise IOError('data not present on disk. '
                          'set download_if_missing=True to download')

        print("downloading DR7 quasar dataset from %s to %s"
              % (DATA_URL, data_home))

        zipped_buf = download_with_progress_bar(DATA_URL, return_buffer=True)
        gzf = GzipFile(fileobj=zipped_buf, mode='rb')
        extracted_buf = BytesIO(gzf.read())
        data = np.loadtxt(extracted_buf,
                          skiprows=SKIP_ROWS,
                          usecols=COLUMN_NUMBERS,
                          dtype=DR7_DTYPE)
        np.save(archive_file, data)

    else:
        data = np.load(archive_file)

    return data
