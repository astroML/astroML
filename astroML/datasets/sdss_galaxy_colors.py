from __future__ import print_function, division

import os

import numpy as np
from . import get_data_home
from .tools import sql_query

SPECCLASS = ['UNKNOWN', 'STAR', 'GALAXY', 'QSO',
             'HIZ_QSO', 'SKY', 'STAR_LATE', 'GAL_EM']

NOBJECTS = 50000

GAL_COLORS_DTYPE = [('u', float),
                    ('g', float),
                    ('r', float),
                    ('i', float),
                    ('z', float),
                    ('specClass', int),
                    ('redshift', float),
                    ('redshift_err', float)]

ARCHIVE_FILE = 'sdss_galaxy_colors.npy'


def fetch_sdss_galaxy_colors(data_home=None, download_if_missing=True):
    """Loader for SDSS galaxy colors.

    This function directly queries the sdss SQL database at
    http://cas.sdss.org/

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
    data : recarray, shape = (10000,)
        record array containing magnitudes and redshift for each galaxy
    """
    data_home = get_data_home(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    archive_file = os.path.join(data_home, ARCHIVE_FILE)

    query_text = ('\n'.join(
            ("SELECT TOP %i" % NOBJECTS,
             "   p.u, p.g, p.r, p.i, p.z, s.specClass, s.z, s.zerr",
             "FROM PhotoObj AS p",
             "   JOIN SpecObj AS s ON s.bestobjid = p.objid",
             "WHERE ",
             "   p.u BETWEEN 0 AND 19.6",
             "   AND p.g BETWEEN 0 AND 20",
             "   AND s.specClass > 1 -- not UNKNOWN or STAR",
             "   AND s.specClass <> 5 -- not SKY",
             "   AND s.specClass <> 6 -- not STAR_LATE")))

    if not os.path.exists(archive_file):
        if not download_if_missing:
            raise IOError('data not present on disk. '
                          'set download_if_missing=True to download')

        print("querying for %i objects" % NOBJECTS)
        print(query_text)
        output = sql_query(query_text)
        print("finished.")

        data = np.loadtxt(output, delimiter=',',
                          skiprows=1, dtype=GAL_COLORS_DTYPE)
        np.save(archive_file, data)

    else:
        data = np.load(archive_file)

    return data
