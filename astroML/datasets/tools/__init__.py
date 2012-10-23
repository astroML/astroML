"""
tools for the dataset loaders
"""

from .download import download_with_progress_bar
from .sql_query import sql_query
from .cas_query import *
from .sdss_fits import sdss_fits_url, sdss_fits_filename, SDSSfits


def get_data_home(data_home=None):
    """Get the home data directory.

    By default the data dir is set to a folder named 'astroML_data'
    in the user home folder.

    Alternatively, it can be set by the 'ASTROML_DATA' environment
    variable or programatically by giving an explit folder path. The
    '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.
    """
    import os
    if data_home is None:
        data_home = os.environ.get('ASTROML_DATA',
                                   os.path.join('~', 'astroML_data'))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home
