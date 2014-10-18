import os
import tarfile

import numpy as np

from . import get_data_home
from .tools import download_with_progress_bar

DATA_URL = ("http://www.astro.washington.edu/users/bsesar/"
            "S82_RRLyr/RRLyr_ugriz_templates.tar.gz")


def fetch_rrlyrae_templates(data_home=None, download_if_missing=True):
    """Loader for RR-Lyrae template data

    These are the light-curve templates from Sesar et al 2010, ApJ 708:717

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
    data : numpy record array
        record array containing the templates
    """
    data_home = get_data_home(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    data_file = os.path.join(data_home, os.path.basename(DATA_URL))

    if not os.path.exists(data_file):
        if not download_if_missing:
            raise IOError('data not present on disk. '
                          'set download_if_missing=True to download')

        databuffer = download_with_progress_bar(DATA_URL)
        open(data_file, 'wb').write(databuffer)

    data = tarfile.open(data_file)

    return dict([(name.strip('.dat'),
                  np.loadtxt(data.extractfile(name)))
                 for name in data.getnames()])
