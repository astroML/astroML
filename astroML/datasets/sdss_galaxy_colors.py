import os

import numpy as np
from . import get_data_home
from .tools import sql_query

NOBJECTS = 50000

GAL_COLORS_NAMES = ['u', 'g', 'r', 'i', 'z', 'specClass',
                    'redshift', 'redshift_err']

ARCHIVE_FILE = 'sdss_galaxy_colors.npy'


def fetch_sdss_galaxy_colors(data_home=None, download_if_missing=True):
    """Loader for SDSS galaxy colors.

    This function directly queries the sdss SQL database at
    http://cas.sdss.org/

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
    data : recarray, shape = (10000,)
        record array containing magnitudes and redshift for each galaxy
    """
    data_home = get_data_home(data_home)

    archive_file = os.path.join(data_home, ARCHIVE_FILE)

    query_text = ('\n'.join(("SELECT TOP %i" % NOBJECTS,
                             "  p.u, p.g, p.r, p.i, p.z, s.class, s.z, s.zerr",
                             "FROM PhotoObj AS p",
                             "  JOIN SpecObj AS s ON s.bestobjid = p.objid",
                             "WHERE ",
                             "  p.u BETWEEN 0 AND 19.6",
                             "  AND p.g BETWEEN 0 AND 20",
                             "  AND s.class <> 'UNKNOWN'",
                             "  AND s.class <> 'STAR'",
                             "  AND s.class <> 'SKY'",
                             "  AND s.class <> 'STAR_LATE'")))

    if not os.path.exists(archive_file):
        if not download_if_missing:
            raise IOError('data not present on disk. '
                          'set download_if_missing=True to download')

        print("querying for %i objects" % NOBJECTS)
        print(query_text)
        output = sql_query(query_text)
        print("finished.")

        kwargs = {'delimiter': ',', 'skip_header': 2,
                  'names': GAL_COLORS_NAMES, 'dtype': None,
                  'encoding': 'ascii',
                  }

        data = np.genfromtxt(output, **kwargs)
        np.save(archive_file, data)

    else:
        data = np.load(archive_file)

    return data
