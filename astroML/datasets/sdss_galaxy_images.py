import os
from io import BytesIO
from gzip import GzipFile

import numpy as np

from astroML.datasets import get_data_home
from astroML.datasets.tools import download_with_progress_bar

IMAGES_URL = ('https://github.com/astroML/astroML-data/raw/main/datasets/'
              'sdss_images_1000.npy.gz')
LABELS_URL = ('https://github.com/astroML/astroML-data/raw/main/datasets/'
              'sdss_labels_1000.npy')


def fetch_sdss_galaxy_images(data_home=None, download_if_missing=True):
    """
    Loader for SDSS galaxy images.

    A sample of 1000 coloured galaxy image stamps are loaded along with
    labels for their morphological classification.

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
    data : ndarray, shape = (1000, 68, 68, 3)
        Array containing image data for 1000 galaxies in 3 colours.

    labels: ndarray, shape = (1000,)
        Labels of morphological classification (1 for spiral, 0 for elliptical).

    Notes
    -----
    The sample selection is courtesy of Marc Huertas-Company from the
    full dataset of Nair & Abraham 2010 ApJS 186:427.
    """

    data_home = get_data_home(data_home)

    images_file = os.path.join(data_home, os.path.basename(IMAGES_URL).split('.gz')[0])
    labels_file = os.path.join(data_home, os.path.basename(LABELS_URL))

    if not os.path.exists(images_file) or not os.path.exists(labels_file):
        if not download_if_missing:
            raise IOError('data not present on disk. '
                          'set download_if_missing=True to download')

        zipped_buf = download_with_progress_bar(IMAGES_URL, return_buffer=True)
        gzf = GzipFile(fileobj=zipped_buf, mode='rb')
        data = np.load(BytesIO(gzf.read()))
        np.save(images_file, data)
        labels_buffer = download_with_progress_bar(LABELS_URL, return_buffer=True)
        labels = np.load(labels_buffer)
        np.save(labels_file, labels)
    else:
        data = np.load(images_file)
        labels = np.load(labels_file)

    return data, labels
