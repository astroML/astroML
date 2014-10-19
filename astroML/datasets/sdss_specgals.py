from __future__ import print_function, division

import os

import numpy as np

from . import get_data_home
from .tools import download_with_progress_bar

DATA_URL = ("http://www.astro.washington.edu/users/ivezic/"
            "DMbook/data/SDSSspecgalsDR8.fit")


def fetch_sdss_specgals(data_home=None, download_if_missing=True):
    """Loader for SDSS Galaxies with spectral information

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
    data : recarray, shape = (327260,)
        record array containing pipeline parameters

    Notes
    -----
    These were compiled from the SDSS database using the following SQL query::

        SELECT
          G.ra, G.dec, S.mjd, S.plate, S.fiberID, --- basic identifiers
          --- basic spectral data
          S.z, S.zErr, S.rChi2, S.velDisp, S.velDispErr,
          --- some useful imaging parameters
          G.extinction_r, G.petroMag_r, G.psfMag_r, G.psfMagErr_r,
          G.modelMag_u, modelMagErr_u, G.modelMag_g, modelMagErr_g,
          G.modelMag_r, modelMagErr_r, G.modelMag_i, modelMagErr_i,
          G.modelMag_z, modelMagErr_z, G.petroR50_r, G.petroR90_r,
          --- line fluxes for BPT diagram and other derived spec. parameters
          GSL.nii_6584_flux, GSL.nii_6584_flux_err, GSL.h_alpha_flux,
          GSL.h_alpha_flux_err, GSL.oiii_5007_flux, GSL.oiii_5007_flux_err,
          GSL.h_beta_flux, GSL.h_beta_flux_err, GSL.h_delta_flux,
          GSL.h_delta_flux_err, GSX.d4000, GSX.d4000_err, GSE.bptclass,
          GSE.lgm_tot_p50, GSE.sfr_tot_p50, G.objID, GSI.specObjID
        INTO mydb.SDSSspecgalsDR8 FROM SpecObj S CROSS APPLY
          dbo.fGetNearestObjEQ(S.ra, S.dec, 0.06) N, Galaxy G,
          GalSpecInfo GSI, GalSpecLine GSL, GalSpecIndx GSX, GalSpecExtra GSE
        WHERE N.objID = G.objID
          AND GSI.specObjID = S.specObjID
          AND GSL.specObjID = S.specObjID
          AND GSX.specObjID = S.specObjID
          AND GSE.specObjID = S.specObjID
          --- add some quality cuts to get rid of obviously bad measurements
          AND (G.petroMag_r > 10 AND G.petroMag_r < 18)
          AND (G.modelMag_u-G.modelMag_r) > 0
          AND (G.modelMag_u-G.modelMag_r) < 6
          AND (modelMag_u > 10 AND modelMag_u < 25)
          AND (modelMag_g > 10 AND modelMag_g < 25)
          AND (modelMag_r > 10 AND modelMag_r < 25)
          AND (modelMag_i > 10 AND modelMag_i < 25)
          AND (modelMag_z > 10 AND modelMag_z < 25)
          AND S.rChi2 < 2
          AND (S.zErr > 0 AND S.zErr < 0.01)
          AND S.z > 0.02
          --- end of query ---

    Examples
    --------
    >>> from astroML.datasets import fetch_sdss_specgals
    >>> data = fetch_sdss_specgals()
    >>> data.shape  # number of objects in dataset
    (661598,)
    >>> data.names[:5]  # first five column names
    ['ra', 'dec', 'mjd', 'plate', 'fiberID']
    >>> print(data['ra'][:3])  # first three RA values
    [ 146.71419105  146.74414186  146.62857334]
    >>> print(data['dec'][:3])  #  first three declination values
    [-1.04127639 -0.6522198  -0.7651468 ]
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


def fetch_great_wall(data_home=None, download_if_missing=True,
                     xlim=(-375, -175), ylim=(-300, 200)):
    """Get the 2D SDSS "Great Wall" distribution, following Cowan et al 2008

    Parameters
    ----------
    data_home : optional, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit learn data is stored in '~/astroML_data' subfolders.

    download_if_missing : optional, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    xlim, ylim : tuples or None
        the limits in Mpc of the data: default values are the same as that
        used for the plots in Cowan 2008.  If set to None, no cuts will
        be performed.

    Returns
    -------
    data : ndarray, shape = (Ngals, 2)
        grid of projected (x, y) locations of galaxies in Mpc
    """
    # local imports so we don't need dependencies for loading module
    from scipy.interpolate import interp1d
    from ..cosmology import Cosmology

    data = fetch_sdss_specgals(data_home, download_if_missing)

    # cut to the part of the sky with the "great wall"
    data = data[(data['dec'] > -7) & (data['dec'] < 7)]
    data = data[(data['ra'] > 80) & (data['ra'] < 280)]

    # do a redshift cut, following Cowan et al 2008
    z = data['z']
    data = data[(z > 0.01) & (z < 0.12)]

    # use redshift to compute absolute r-band magnitude
    cosmo = Cosmology(omegaM=0.27, omegaL=0.73, h=0.732)

    # first sample the distance modulus on a grid
    zgrid = np.linspace(min(data['z']), max(data['z']), 100)
    mugrid = np.array([cosmo.mu(z) for z in zgrid])
    f = interp1d(zgrid, mugrid)
    mu = f(data['z'])

    # do an absolute magnitude cut at -20
    Mr = data['petroMag_r'] + data['extinction_r'] - mu
    data = data[Mr < -21]

    # compute distances in the equatorial plane
    # first sample comoving distance
    Dcgrid = np.array([cosmo.Dc(z) for z in zgrid])
    f = interp1d(zgrid, Dcgrid)
    dist = f(data['z'])

    locs = np.vstack([dist * np.cos(data['ra'] * np.pi / 180.),
                      dist * np.sin(data['ra'] * np.pi / 180.)]).T

    # cut on x and y limits if specified
    if xlim is not None:
        locs = locs[(locs[:, 0] > xlim[0]) & (locs[:, 0] < xlim[1])]
    if ylim is not None:
        locs = locs[(locs[:, 1] > ylim[0]) & (locs[:, 1] < ylim[1])]

    return locs
