"""
Fetch the LIGO BigDog time-domain dataset
"""
from __future__ import print_function, division

import os
from ..py3k_compat import BytesIO
from gzip import GzipFile
import numpy as np

from . import get_data_home
from .tools import download_with_progress_bar

DATA_URL_LARGE = ('http://www.astro.washington.edu/users/ivezic/'
                  'DMbook/LIGO/hoft.968653908-968655956.H1.dat.gz')
LOCAL_FILE_LARGE = 'LIGO_large.npy'

DATA_URL = 'http://www.ligo.org/science/GW100916/HLV-strain.txt'
LOCAL_FILE = 'LIGO_bigdog.npy'


def fetch_LIGO_large(data_home=None, download_if_missing=True):
    """Loader for LIGO large dataset

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
    data : ndarray
    dt : float
        data represents ~2000s of amplitude data from LIGO hanford;
        dt is the time spacing between measurements in seconds.
    """
    data_home = get_data_home(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    local_file = os.path.join(data_home, LOCAL_FILE_LARGE)

    if os.path.exists(local_file):
        data = np.load(local_file)

    else:
        if not download_if_missing:
            raise IOError('data not present on disk. '
                          'set download_if_missing=True to download')

        print("downloading LIGO bigdog data from %s to %s"
              % (DATA_URL_LARGE, local_file))

        zipped_buf = download_with_progress_bar(DATA_URL_LARGE,
                                                return_buffer=True)
        gzf = GzipFile(fileobj=zipped_buf, mode='rb')
        print("uncompressing file...")
        extracted_buf = BytesIO(gzf.read())
        data = np.loadtxt(extracted_buf)
        np.save(local_file, data)

    return data, 1. / 4096


def fetch_LIGO_bigdog(data_home=None, download_if_missing=True):
    """Loader for LIGO bigdog event

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
    data : record array
        The data is 10 seconds of measurements from three sites, along with
        the time of each measurement.

    Examples
    --------
    >>> from astroML.datasets import fetch_LIGO_bigdog
    >>> data = fetch_LIGO_bigdog()
    >>> print(data.dtype.names)
    ('t', 'Hanford', 'Livingston', 'Virgo')
    >>> print(data['t'][:3])
    [  0.00000000e+00   6.10400000e-05   1.22070000e-04]
    >>> print(data['Hanford'][:3])
    [  1.26329846e-17   1.26846778e-17   1.19187381e-17]
    """
    data_home = get_data_home(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    local_file = os.path.join(data_home, LOCAL_FILE)

    if os.path.exists(local_file):
        data = np.load(local_file)

    else:
        if not download_if_missing:
            raise IOError('data not present on disk. '
                          'set download_if_missing=True to download')

        print("downloading LIGO bigdog data from %s to %s"
              % (DATA_URL, local_file))

        buffer = download_with_progress_bar(DATA_URL, return_buffer=True)
        data = np.loadtxt(buffer, skiprows=2,
                          dtype=[('t', 'f8'),
                                 ('Hanford', 'f8'),
                                 ('Livingston', 'f8'),
                                 ('Virgo', 'f8')])
        np.save(local_file, data)

    return data
