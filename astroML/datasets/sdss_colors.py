import os
import sys
import urllib2

import numpy as np

from .tools import get_data_home

SDSS_COLORS_TRAIN = 'sdss_colors_train.npz'
SDSS_COLORS_TEST = 'sdss_colors_test.npz'

SDSS_COLORS_TOPLEVEL_URL = "http://www.astro.washington.edu/users/ajc/teaching/a597/"
SDSS_COLORS_TRAIN_URL = os.path.join(SDSS_COLORS_TOPLEVEL_URL,
                                     'homework/homework1/sdssdr6_colors_class_train.dat')
SDSS_COLORS_TEST_URL = os.path.join(SDSS_COLORS_TOPLEVEL_URL, 
                                    'homework/homework1/sdssdr6_colors_class.200000.dat')

def fetch_sdss_colors_train(data_home=None, shuffle=False,
                            download_if_missing=True, random_state=0):
    """Loader for SDSS photometric data

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all astroML data is stored in '~/astroML_data' subfolders.

    shuffle : boolean, optional, default: False
        If True, then shuffle the data using the specified random state

    download_if_missing: boolean, optional, default: True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    random_state : optional, integer or RandomState object, default: 0
        The seed or RandomState generator used to shuffle the data
        Unused if ``shuffle == False``.

    Notes
    ------
    The data consists of 2 x 10^5 mixed observations in u-g, g-r, r-i, i-z,
    with spectroscopic redshifts.  Objects with zero redshift are stars,
    objects with nonzero redshift are quasars.
    """
    data_home = get_data_home(data_home)

    target_file = os.path.join(data_home, SDSS_COLORS_TRAIN)

    if not os.path.exists(target_file):
        if not download_if_missing:
            raise IOError("SDSS colors training data not found")
        print ("downloading sdss colors training data from %s to %s" 
               % (SDSS_COLORS_TRAIN_URL, target_file))

        # create a password manager
        password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()

        # Add the username and password.
        # If we knew the realm, we could use it instead of None.
        password_mgr.add_password(None, SDSS_COLORS_TOPLEVEL_URL,
                                  'teaching', 'TeachMe')
        handler = urllib2.HTTPBasicAuthHandler(password_mgr)

        # create "opener" (OpenerDirector instance)
        opener = urllib2.build_opener(handler)

        # use the opener to fetch a URL
        fhandle = opener.open(SDSS_COLORS_TRAIN_URL)

        X = np.loadtxt(fhandle)
        data = X[:, :4]
        redshifts = X[:, 4]
        np.savez(target_file, data=data, redshifts=redshifts)

    else:
        file_obj = np.load(target_file)
        data = file_obj['data']
        redshifts = file_obj['redshifts']

    labels = (redshifts < 0.01).astype(int)

    return data, labels


def fetch_sdss_colors_test(data_home=None, shuffle=False,
                           download_if_missing=True, random_state=0):
    """Loader for SDSS photometric data

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all astroML data is stored in '~/astroML_data' subfolders.

    shuffle : boolean, optional, default: False
        If True, then shuffle the data using the specified random state

    download_if_missing: boolean, optional, default: True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    random_state : optional, integer or RandomState object, default: 0
        The seed or RandomState generator used to shuffle the data
        Unused if ``shuffle == False``.

    Notes
    ------
    The data consists of 2 x 10^5 mixed observations in u-g, g-r, r-i, i-z,
    with spectroscopic redshifts.  Objects with zero redshift are stars,
    objects with nonzero redshift are quasars.
    """
    data_home = get_data_home(data_home)

    target_file = os.path.join(data_home, SDSS_COLORS_TEST)

    if not os.path.exists(target_file):
        if not download_if_missing:
            raise IOError("SDSS colors test data not found")
        print ("downloading sdss colors test data from %s to %s" 
               % (SDSS_COLORS_TEST_URL, target_file))

        # create a password manager
        password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()

        # Add the username and password.
        # If we knew the realm, we could use it instead of None.
        password_mgr.add_password(None, SDSS_COLORS_TOPLEVEL_URL,
                                  'teaching', 'TeachMe')
        handler = urllib2.HTTPBasicAuthHandler(password_mgr)

        # create "opener" (OpenerDirector instance)
        opener = urllib2.build_opener(handler)

        # use the opener to fetch a URL
        fhandle = opener.open(SDSS_COLORS_TEST_URL)

        X = np.loadtxt(fhandle)
        data = X[:, :4]
        labels = X[:, 4].astype(int)
        np.savez(target_file, data=data, labels=labels)

    else:
        file_obj = np.load(target_file)
        data = file_obj['data']
        labels = file_obj['labels']

    return data, labels
