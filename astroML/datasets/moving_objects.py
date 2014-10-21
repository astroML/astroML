from __future__ import print_function, division

import os
from gzip import GzipFile
import numpy as np

from .tools import download_with_progress_bar
from ..py3k_compat import BytesIO
from . import get_data_home

DATA_URL = 'http://www.astro.washington.edu/users/ivezic/sdssmoc/ADR3.dat.gz'

ARCHIVE_FILE = 'moving_objects.npy'

ADR4_dtype = [('moID', 'a6'),
              ('sdss_run', 'i4'),
              ('sdss_col', 'i4'),
              ('sdss_field', 'i4'),
              ('sdss_obj', 'i4'),
              ('rowc', 'f4'),
              ('colc', 'f4'),
              ('mjd', 'f8'),
              ('ra', 'f8'),
              ('dec', 'f8'),
              ('lambda', 'f8'),
              ('beta', 'f8'),
              ('phi', 'f8'),
              ('vmu', 'f4'),
              ('vmu_err', 'f4'),
              ('vnu', 'f4'),
              ('vnu_err', 'f4'),
              ('vlambda', 'f4'),
              ('vbeta', 'f4'),
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
              ('mag_a', 'f4'),
              ('err_a', 'f4'),
              ('mag_V', 'f4'),
              ('mag_B', 'f4'),
              ('ast_flag', 'i4'),
              ('ast_num', 'i8'),
              ('ast_designation', 'a17'),
              ('ast_det_count', 'i4'),
              ('ast_det_total', 'i4'),
              ('ast_flags', 'i8'),
              ('ra_comp', 'f8'),
              ('dec_comp', 'f8'),
              ('mag_comp', 'f4'),
              ('r_helio', 'f4'),
              ('r_geo', 'f4'),
              ('phase', 'f4'),
              ('cat_id', 'a15'),
              ('H', 'f4'),
              ('G', 'f4'),
              ('Arc', 'f4'),
              ('Epoch', 'f8'),
              ('a', 'f8'),
              ('e', 'f8'),
              ('i', 'f8'),
              ('asc_node', 'f8'),
              ('arg_peri', 'f8'),
              ('M', 'f8'),
              ('PEcat_id', 'a17'),
              ('aprime', 'f8'),
              ('eprime', 'f8'),
              ('sin_iprime', 'f8')]


def fetch_moving_objects(data_home=None, download_if_missing=True,
                         Parker2008_cuts=False):
    """Loader for SDSS moving objects datasets

    Parameters
    ----------
    data_home : optional, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit learn data is stored in '~/astroML_data' subfolders.

    download_if_missing : optional, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Parker2008_cuts : bool (optional)
        If true, apply cuts on magnitudes and orbital parameters used in
        Parker et al. 2008

    Returns
    -------
    data : recarray, shape = (??,)
        record array containing 60 values for each item

    Notes
    -----
    See http://www.astro.washington.edu/users/ivezic/sdssmoc/sdssmoc3.html
    Columns 0, 35, 45, and 56 are left out of the fetch: they are string
    parameters.  Only columns with known orbital parameters are saved.

    Examples
    --------
    >>> data = fetch_moving_objects()
    >>> print(len(data))  # number of objects
    104686
    >>> u_g = data['mag_u'] - data['mag_g']
    >>> print(u_g[:5])  # first five u-g colors of the dataset
    [ 1.48999977  1.80000114  1.78000069  1.65000153  2.01000023]
    """
    data_home = get_data_home(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    archive_file = os.path.join(data_home, ARCHIVE_FILE)

    if not os.path.exists(archive_file):
        if not download_if_missing:
            raise IOError('data not present on disk. '
                          'set download_if_missing=True to download')

        print("downloading moving object catalog from %s to %s"
              % (DATA_URL, data_home))

        zipped_buf = download_with_progress_bar(DATA_URL, return_buffer=True)
        gzf = GzipFile(fileobj=zipped_buf, mode='rb')
        print("uncompressing file...")
        extracted_buf = BytesIO(gzf.read())
        data = np.loadtxt(extracted_buf, dtype=ADR4_dtype)

        # Select unique sources with known orbital elements
        flag = (data['ast_flag'] == 1) & (data['ast_det_count'] == 1)
        data = data[flag]

        np.save(archive_file, data)

    else:
        data = np.load(archive_file)

    if Parker2008_cuts:
        i_z = data['mag_i'] - data['mag_z']

        flag = ((data['aprime'] >= 0.01) & (data['aprime'] <= 100) &
                (data['mag_a'] <= 0.4) & (data['mag_a'] >= -0.3) &
                (i_z <= 0.6) & (i_z >= -0.8))

        data = data[flag]

    return data
