import os

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table, vstack

from . import get_data_home

# We store the data in two parts to comply with GitHub 100Mb file size limit
DATA_URL1 = ("https://github.com/astroML/astroML-data/raw/master/datasets/"
             "SDSSspecgalsDR8_1.fit.gz")
DATA_URL2 = ("https://github.com/astroML/astroML-data/raw/master/datasets/"
             "SDSSspecgalsDR8_2.fit.gz")


def fetch_sdss_specgals(data_home=None, download_if_missing=True):
    """Loader for SDSS Galaxies with spectral information

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
    data : recarray, shape = (661598,)
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
    >>> data = fetch_sdss_specgals()  # doctest: +IGNORE_OUTPUT +REMOTE_DATA
    >>> # number of objects in dataset
    >>> data.shape  # doctest: +REMOTE_DATA
    (661598,)
    >>> # first five column names
    >>> data.dtype.names[:5]  # doctest: +REMOTE_DATA
    ('ra', 'dec', 'mjd', 'plate', 'fiberID')
    >>> # first three RA values
    >>> print(data['ra'][:3])  # doctest: +REMOTE_DATA
    [ 146.71419105  146.74414186  146.62857334]
    >>> # first three declination values
    >>> print(data['dec'][:3])  # doctest: +REMOTE_DATA
    [-1.04127639 -0.6522198  -0.7651468 ]
    """

    data_home = get_data_home(data_home)

    archive_file1 = os.path.join(data_home, os.path.basename(DATA_URL1))
    archive_file2 = os.path.join(data_home, os.path.basename(DATA_URL2))

    if not (os.path.exists(archive_file1) and os.path.exists(archive_file2)):
        if not download_if_missing:
            raise IOError('data not present on disk. '
                          'set download_if_missing=True to download')

        for url, name in zip([DATA_URL1, DATA_URL2],
                             [archive_file1, archive_file2]):
            data = Table.read(url)
            data.write(name)

    data1 = Table.read(archive_file1)
    data2 = Table.read(archive_file2)

    data = vstack([data1, data2])
    return np.asarray(data)


def fetch_great_wall(data_home=None, download_if_missing=True,
                     xlim=(-375, -175), ylim=(-300, 200), cosmo=None):
    """Get the 2D SDSS "Great Wall" distribution, following Cowan et al 2008

    Parameters
    ----------
    data_home : optional, default=None
        Specify another download and cache folder for the datasets. By default
        all astroML data is stored in '~/astroML_data'.

    download_if_missing : optional, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    xlim, ylim : tuples or None
        the limits in Mpc of the data: default values are the same as that
        used for the plots in Cowan 2008.  If set to None, no cuts will
        be performed.

    cosmo : `astropy.cosmology` instance specifying cosmology
        to use when generating the sample.  If not provided,
        a Flat Lambda CDM model with H0=73.2, Om0=0.27, Tcmb0=0 is used.

    Returns
    -------
    data : ndarray, shape = (Ngals, 2)
        grid of projected (x, y) locations of galaxies in Mpc
    """
    # local imports so we don't need dependencies for loading module
    from scipy.interpolate import interp1d

    # We need some cosmological information to compute the r-band
    #  absolute magnitudes.
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=73.2, Om0=0.27, Tcmb0=0)

    data = fetch_sdss_specgals(data_home, download_if_missing)

    # cut to the part of the sky with the "great wall"
    data = data[(data['dec'] > -7) & (data['dec'] < 7)]
    data = data[(data['ra'] > 80) & (data['ra'] < 280)]

    # do a redshift cut, following Cowan et al 2008
    z = data['z']
    data = data[(z > 0.01) & (z < 0.12)]

    # first sample the distance modulus on a grid
    zgrid = np.linspace(min(data['z']), max(data['z']), 100)
    mugrid = cosmo.distmod(zgrid).value
    f = interp1d(zgrid, mugrid)
    mu = f(data['z'])

    # do an absolute magnitude cut at -20
    Mr = data['petroMag_r'] + data['extinction_r'] - mu
    data = data[Mr < -21]

    # compute distances in the equatorial plane
    # first sample comoving distance
    Dcgrid = cosmo.comoving_distance(zgrid).value
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
