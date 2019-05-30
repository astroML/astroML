"""
NASA Sloan Atlas dataset size reduction
---------------------------------------

The NASA Sloan Atlas dataset is contained in a ~0.5GB available at
http://www.nsatlas.org/data

This function fetches a ~50MB subset of that data.  This subset is created
using the code that can be found at examples/datasets/truncate_nsa_data.py
"""
import os

import numpy as np

from .tools import download_with_progress_bar
from . import get_data_home


DATA_URL = ('https://github.com/astroML/astroML-data/raw/master/datasets/'
            'nsa_v0_1_2_reduced.npy')

ARCHIVE_FILE = os.path.basename(DATA_URL)


def fetch_nasa_atlas(data_home=None,
                     download_if_missing=True):
    """Loader for NASA galaxy atlas data

    Parameters
    ----------
    data_home : optional, default=None
        Specify another download and cache folder for the datasets. By default
        all astroML data is stored in '~/astroML_data'.

    download_if_missing : optional, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    data : ndarray
        The data, in the form of a numpy record array.

    Notes
    -----
    This is the file created by the example script at
        examples/datasets/truncate_nsa_data.py
    For an explanation of the meaning of the fields, see the description at
        http://www.nsatlas.org/data
    """
    data_home = get_data_home(data_home)

    archive_file = os.path.join(data_home, ARCHIVE_FILE)

    if not os.path.exists(archive_file):
        if not download_if_missing:
            raise IOError('data not present on disk. '
                          'set download_if_missing=True to download')

        print("downloading NASA atlas data from %s to %s"
              % (DATA_URL, data_home))

        buf = download_with_progress_bar(DATA_URL, return_buffer=True)
        data = np.load(buf)

        np.save(archive_file, data)

    else:
        data = np.load(archive_file)

    return data
