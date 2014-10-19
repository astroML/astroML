from __future__ import print_function, division

import os
import numpy as np
from . import get_data_home
from .tools import download_with_progress_bar

DATA_URL = 'http://www.astro.washington.edu/users/vanderplas/spec4000.npz'
ARCHIVE_FILE = 'spec4000.npz'


def reconstruct_spectra(data):
    """Compute the reconstructed spectra.

    Parameters
    ----------
    data: NpzFile
        numpy data object returned by fetch_sdss_corrected_spectra.

    Returns
    -------
    spec_recons: ndarray
        Reconstructed spectra, using principal components to interpolate
        across the masked region.
    """
    spectra = data['spectra']
    coeffs = data['coeffs']
    evecs = data['evecs']
    mask = data['mask']
    mu = data['mu']
    norms = data['norms']

    spec_recons = spectra.copy()

    nev = coeffs.shape[1]

    spec_fill = mu + np.dot(coeffs, evecs[:nev])
    spec_fill *= norms[:, np.newaxis]

    spec_recons[mask] = spec_fill[mask]

    return spec_recons


def compute_wavelengths(data):
    """Compute the wavelength associated with spectra.

    Paramters
    ---------

    Parameters
    ----------
    data: NpzFile
        numpy data object returned by fetch_sdss_corrected_spectra.

    Returns
    -------
    wavelength: ndarray
        One-dimensional wavelength array for spectra.
    """

    return 10 ** (data['coeff0']
                  + data['coeff1'] * np.arange(data['spectra'].shape[1]))


def fetch_sdss_corrected_spectra(data_home=None,
                                 download_if_missing=True):
    """Loader for Iterative PCA pre-processed galaxy spectra

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
    data : NpzFile
        The data dictionary

    Notes
    -----
    This is the file created by the example script
    examples/datasets/compute_sdss_pca.py
    """
    data_home = get_data_home(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    archive_file = os.path.join(data_home, ARCHIVE_FILE)

    if not os.path.exists(archive_file):
        if not download_if_missing:
            raise IOError('data not present on disk. '
                          'set download_if_missing=True to download')

        print("downloading PCA-processed SDSS spectra from %s to %s"
              % (DATA_URL, data_home))

        buf = download_with_progress_bar(DATA_URL, return_buffer=True)
        data = np.load(buf)

        data_dict = dict([(key, data[key]) for key in data.files])
        np.savez(archive_file, **data_dict)

    else:
        data = np.load(archive_file)

    return data
