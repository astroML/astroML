import os
from ..py3k_compat import BytesIO
import tarfile

import numpy as np

from . import get_data_home
from .tools import download_with_progress_bar

TARGETLIST_URL = ("http://www.astro.washington.edu/users/ivezic/"
                  "linear/allDataFinal/allLINEARfinal_targets.dat")
DATA_URL = ("http://www.astro.washington.edu/users/ivezic/"
            "linear/allDataFinal/allLINEARfinal_dat.tar.gz")

# old version of the data
#GENEVA_URL = ("http://www.astro.washington.edu/users/ivezic/"
#              "DMbook/data/LINEARattributes.dat"
#GENEVA_ARCHIVE = 'LINEARattributes.npy'
#ARCHIVE_DTYPE = [(s, 'f8') for s in ('RA', 'Dec', 'ug', 'gi', 'iK',
#                                     'JK', 'logP', 'amp', 'skew')]

GENEVA_URL = ("http://www.astro.washington.edu/users/ivezic/"
              "DMbook/data/LINEARattributesFinalApr2013.dat")
GENEVA_ARCHIVE = 'LINEARattributesFinalApr2013.npy'
ARCHIVE_DTYPE = ([(s, 'f8') for s in ('RA', 'Dec', 'ug', 'gi', 'iK',
                                     'JK', 'logP', 'amp', 'skew',
                                     'kurt', 'magMed', 'nObs')]
                 + [('LCtype', 'i4'), ('LINEARobjectID', '|S20')])


target_names = ['objectID', 'raLIN', 'decLIN', 'raSDSS', 'decSDSS', 'r',
                'ug', 'gr', 'ri', 'iz', 'JK', '<mL>', 'std', 'rms',
                'Lchi2', 'LP1', 'phi1', 'S', 'prior']


class LINEARdata(object):
    """A container class for the linear dataset.

    Because the dataset is often not needed all at once, this class
    offers tools to access just the needed components

    Example
    -------
    >>> data = fetch_LINEAR_sample()
    >>> lightcurve = data[data.ids[0]]
    """
    @staticmethod
    def _name_to_id(name):
        return int(name.split('.')[0])

    @staticmethod
    def _id_to_name(id):
        return str(id) + '.dat'

    def __init__(self, data_file, targetlist_file):
        self.targets = np.recfromtxt(targetlist_file)
        self.targets.dtype.names = target_names

        self.dataF = tarfile.open(data_file)
        self.ids = np.array(list(map(self._name_to_id, self.dataF.getnames())))

        # rearrange targets so lists are in the same order
        self.targets = self.targets[self.targets['objectID'].argsort()]
        ind = self.targets['objectID'].searchsorted(self.ids)
        self.targets = self.targets[ind]

    def get_light_curve(self, id):
        """Get a light curve with the given id.

        Parameters
        ----------
        id: integer
            LINEAR id of the desired object

        Returns
        -------
        lightcurve: ndarray
            a size (n_observations, 3) light-curve.
            columns are [MJD, flux, flux_err]
        """
        return self[id]

    def get_target_parameter(self, id, param):
        """Get a target parameter associated with the given id.

        Parameters
        ----------
        id: integer
            LINEAR id of the desired object
        param: string
            parameter name of the desired object (see below)

        Returns
        -------
        val: scalar
            value of the requested target parameter

        Notes
        -----
        Target parameters are one of the following:

        ['objectID', 'raLIN', 'decLIN', 'raSDSS', 'decSDSS', 'r',
         'ug', 'gr', 'ri', 'iz', 'JK', '<mL>', 'std', 'rms',
         'Lchi2', 'LP1', 'phi1', 'S', 'prior']
        """
        i = np.where(self.targets['objectID'] == id)[0]
        try:
            val = self.targets[param][i[0]]
        except:
            raise KeyError(id)

        return val

    def __getitem__(self, id):
        try:
            lc = np.loadtxt(self.dataF.extractfile(self._id_to_name(id)))
        except:
            raise KeyError(id)

        return lc


def fetch_LINEAR_sample(data_home=None, download_if_missing=True):
    """Loader for LINEAR data sample

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
    data : LINEARdata object
        A custom object which provides access to 7010 selected LINEAR light
        curves.
    """
    data_home = get_data_home(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    targetlist_file = os.path.join(data_home, os.path.basename(TARGETLIST_URL))
    data_file = os.path.join(data_home, os.path.basename(DATA_URL))

    if not os.path.exists(targetlist_file):
        if not download_if_missing:
            raise IOError('data not present on disk. '
                          'set download_if_missing=True to download')

        targets = download_with_progress_bar(TARGETLIST_URL)
        open(targetlist_file, 'wb').write(targets)

    if not os.path.exists(data_file):
        if not download_if_missing:
            raise IOError('data not present on disk. '
                          'set download_if_missing=True to download')

        databuffer = download_with_progress_bar(DATA_URL)
        open(data_file, 'wb').write(databuffer)

    return LINEARdata(data_file, targetlist_file)


def fetch_LINEAR_geneva(data_home=None, download_if_missing=True):
    """Loader for LINEAR geneva data.

    This supplements the LINEAR data above with well-determined periods
    and other light curve characteristics.

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
        data on 7000+ LINEAR stars from the Geneva catalog
    """
    data_home = get_data_home(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    archive_file = os.path.join(data_home, GENEVA_ARCHIVE)

    if not os.path.exists(archive_file):
        if not download_if_missing:
            raise IOError('data not present on disk. '
                          'set download_if_missing=True to download')

        databuffer = download_with_progress_bar(GENEVA_URL)
        data = np.loadtxt(BytesIO(databuffer), dtype=ARCHIVE_DTYPE)
        np.save(archive_file, data)
    else:
        data = np.load(archive_file)

    return data
