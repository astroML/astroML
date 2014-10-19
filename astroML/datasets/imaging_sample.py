from __future__ import print_function, division

import os

import numpy as np

from . import get_data_home
from .tools import download_with_progress_bar

DATA_URL = ("http://www.astro.washington.edu/users/"
            "ivezic/DMbook/data/imagingSample_20sqdeg.fit")
DATA_URL = ("http://www.astro.washington.edu/users/"
            "ivezic/DMbook/data/sgSDSSimagingSample.fit")


def fetch_imaging_sample(data_home=None, download_if_missing=True):
    """Loader for SDSS Imaging sample data

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
    data : recarray, shape = (330753,)
        record array containing imaging data

    Examples
    --------
    >>> from astroML.datasets import fetch_imaging_sample
    >>> data = fetch_imaging_sample()
    >>> data.shape  # number of objects in dataset
    (330753,)
    >>> print(data.names[:5])  # names of the first five columns
    ['ra', 'dec', 'run', 'rExtSFD', 'uRaw']
    >>> print(data['ra'][:2])
    [ 0.265165  0.265413]
    >>> print(data['dec'][:2])
    [-0.444861 -0.62201 ]

    Notes
    -----
    This data was selected from the SDSS database using the following SQL
    query::

        SELECT
          round(p.ra,6) as ra, round(p.dec,6) as dec,
          p.run,                              --- comments are preceded by ---
          round(p.extinction_r,3) as rExtSFD, --- r band extinction from SFD
          round(p.modelMag_u,3) as uRaw,      --- ISM-uncorrected model mags
          round(p.modelMag_g,3) as gRaw,      --- rounding up model magnitudes
          round(p.modelMag_r,3) as rRaw,
          round(p.modelMag_i,3) as iRaw,
          round(p.modelMag_z,3) as zRaw,
          round(p.modelMagErr_u,3) as uErr,   --- errors are important!
          round(p.modelMagErr_g,3) as gErr,
          round(p.modelMagErr_r,3) as rErr,
          round(p.modelMagErr_i,3) as iErr,
          round(p.modelMagErr_z,3) as zErr,
          round(p.psfMag_u,3) as psfRaw,      --- psf magnitudes
          round(p.psfMag_g,3) as psfRaw,
          round(p.psfMag_r,3) as psfRaw,
          round(p.psfMag_i,3) as psfRaw,
          round(p.psfMag_z,3) as psfRaw,
          round(p.psfMagErr_u,3) as psfuErr,
          round(p.psfMagErr_g,3) as psfgErr,
          round(p.psfMagErr_r,3) as psfrErr,
          round(p.psfMagErr_i,3) as psfiErr,
          round(p.psfMagErr_z,3) as psfzErr,
          p.type,                   --- tells if a source is resolved or not
          (case when (p.flags & '16') = 0 then 1 else 0 end) as ISOLATED
        INTO mydb.SDSSimagingSample
        FROM PhotoTag p
        WHERE
            --- 10x2 sq.deg.
          p.ra > 0.0 and p.ra < 10.0 and p.dec > -1 and p.dec < 1
            --- resolved and unresolved sources
          and (p.type = 3 OR p.type = 6) and
            --- '4295229440' is magic code for no
            --- DEBLENDED_AS_MOVING or SATURATED objects
          (p.flags & '4295229440') = 0 and
            --- PRIMARY objects only, which implies
            --- !BRIGHT && (!BLENDED || NODEBLEND || nchild == 0)]
          p.mode = 1 and
            --- adopted faint limit (same as about SDSS limit)
          p.modelMag_r < 22.5
        --- the end of query
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
