"""
Tools to download and process SDSS fits files.

More information can be found at
http://www.sdss.org/dr7/products/spectra/index.html
"""
import gc  # garbage collection
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d, uniform_filter1d
from scipy import interpolate

from . import download_with_progress_bar

# This is the URL of the sdss fits spectra
FITS_FILENAME = 'spSpec-%(mjd)05i-%(plate)04i-%(fiber)03i.fit'
SDSS_URL = ('http://das.sdss.org/spectro/1d_26/%(plate)04i/'
            '1d/spSpec-%(mjd)05i-%(plate)04i-%(fiber)03i.fit')

# lines used to generate line-index labeling
LINES = dict(Ha=6564.61,
             Hb=4862.68,
             OI=6302.05,
             OIII=5008.24,
             NIIa=6549.86,
             NIIb=6585.27,
             SIIa=6718.29,
             SIIb=6732.67)


def sdss_fits_url(plate, mjd, fiber):
    """Return the URL of the spectrum FITS file"""
    return SDSS_URL % dict(plate=plate, mjd=mjd, fiber=fiber)


def sdss_fits_filename(plate, mjd, fiber):
    """Return the name of the spectrum FITS file"""
    return FITS_FILENAME % dict(plate=plate, mjd=mjd, fiber=fiber)


spec_cln_dict = ['SPEC_UNKNOWN',
                 'SPEC_STAR',
                 'SPEC_GALAXY',
                 'SPEC_QSO',
                 'SPEC_HIZ_QSO',  # high redshift QSO, z>2.3
                 'SPEC_SKY',
                 'STAR_LATE',  # Type M or later (molecular bands dominate)
                 'GAL_EM']  # emission line galaxy


class SDSSfits(object):
    """A class to open and interact with fits files from SDSS

    Parameters
    ----------
    buf : string or file buffer (optional)
        file path, buffer, or url of SDSS spectra fits file
        if None, then initialize an empty instance.

    Notes
    -----
    This class only provides access to a subset of the information available
    in the sdss spectra fits file.  The raw fits data can be accessed using
    the fits object directly.  This can be found in the attribute
    ``hdulist``.  For details, please refer to the data description:
    http://www.sdss.org/dr7/dm/flatFiles/spSpec.html
    """
    def __init__(self, source=None):
        if source is None:
            pass
        elif isinstance(source, str):
            if source.startswith('http://'):
                self._load_fits_url(source)
            else:
                self._load_fits_file(source)
        else:
            self._load_fits_file(source)

    def _load_fits_url(self, url):
        # fits is an optional dependency: don't import globally
        from astropy.io import fits
        buffer = download_with_progress_bar(url, return_buffer=True)
        self._initialize(fits.open(buffer))

    def _load_fits_file(self, file_or_buffer):
        # fits is an optional dependency: don't import globally
        from astropy.io import fits
        self._initialize(fits.open(file_or_buffer))

    def _initialize(self, hdulist):
        data = hdulist[0].data

        self.name = hdulist[0].header['NAME']
        self.spec_cln = hdulist[0].header['SPEC_CLN']
        self.coeff0 = hdulist[0].header['COEFF0']
        self.coeff1 = hdulist[0].header['COEFF1']
        self.z = hdulist[0].header['Z']
        self.zerr = hdulist[0].header['Z_ERR']
        self.zconf = hdulist[0].header['Z_CONF']

        self.spectrum = data[0]
        self.spectrum_cont = data[1]
        self.error = data[2]
        self.mask = data[3]
        self.large_err = self.error.max() * 2
        self.hdulist = hdulist

    def get_line_ew(self, wavelength):
        i = np.where(abs(self.hdulist[2].data['restWave'] - wavelength) < 1)
        return self.hdulist[2].data['ew'][i]

    def __del__(self):
        if hasattr(self, 'hdulist'):
            del self.hdulist
            gc.collect()

    def copy(self):
        snew = self.__class__()
        for param in ['name', 'spec_cln', 'coeff0', 'coeff1',
                      'z', 'zerr', 'zconf', 'spectrum', 'spectrum_cont',
                      'error', 'large_err', 'mask', 'hdulist']:
            setattr(snew, param, getattr(self, param))
        return snew

    def restframe(self):
        snew = self.copy()
        snew.coeff0 = self.coeff0_restframe()
        snew.z = 0
        return snew

    def __len__(self):
        return len(self.spectrum)

    def log_w_min(self, i=None):
        """
        if i is specified, return log_w_min of bin i
        otherwise, return log_w_min of the spectrum
        """
        if i is None:
            i = 0
        return self.coeff0 + (i - 0.5) * self.coeff1

    def log_w_max(self, i=None):
        """
        if i is specified, return log_w_max of bin i
        otherwise, return log_max of the spectrum
        """
        if i is None:
            i = len(self) - 1
        return self.coeff0 + (i + 0.5) * self.coeff1

    def w_min(self, i=None):
        return 10 ** self.log_w_min(i)

    def w_max(self, i=None):
        return 10 ** self.log_w_max(i)

    def coeff0_restframe(self):
        return self.coeff0 - np.log10(1 + self.z)

    def wavelength(self, restframe=False):
        """
        return the wavelength of the spectrum in angstroms
        """
        if restframe:
            coeff0 = self.coeff0_restframe()
        else:
            coeff0 = self.coeff0
        return 10 ** (coeff0 + self.coeff1 * np.arange(len(self.spectrum)))

    def compute_mask(self, frac=0.5, filtwidth=5):
        """
        return a mask showing where noise spikes to frac over the local
        background
        """
        smoothed_noise = gaussian_filter1d(self.error, filtwidth)
        mask = ((self.error >= (1 + frac) * smoothed_noise)
                | (self.error <= 0)
                | (self.error >= self.large_err)
                | (self.spectrum == 0))
        mask_filtered = uniform_filter1d(mask.astype(float),
                                         max(3, filtwidth))
        return mask_filtered > 0.5 / filtwidth

    def rebin(self, rebin_coeff0, rebin_coeff1, rebin_length):
        """Rebin the spectrum to a new grid.

        Parameters
        ----------
        rebin_coeff0: float
             log minimum wavelength
        rebin_coeff1: float
             log wavelength bin width
        rebin_length: int
             number of bins

        Returns
        -------
        S_new: SDSSfits object
            The new spectrum, rebinned to the desired wavelength binning
        """
        snew = self.copy()
        snew.spectrum = np.zeros(rebin_length)
        snew.error = np.zeros(rebin_length)
        snew.coeff0 = rebin_coeff0
        snew.coeff1 = rebin_coeff1

        N_old = len(self.spectrum)
        N_new = len(snew.spectrum)

        log_w_old = self.coeff0 + (np.arange(N_old + 1) - 0.5) * self.coeff1
        log_w_new = snew.coeff0 + (np.arange(N_new + 1) - 0.5) * snew.coeff1

        # Perform the interpolation.  We'll interpolate the cumulative sum
        #  so that the total flux of the spectrum is conserved.

        # interpolate spectrum
        spec_cuml_old = self.spectrum.cumsum()
        tck = interpolate.splrep(log_w_old, np.hstack(([0], spec_cuml_old)))
        spec_cuml_new = interpolate.splev(log_w_new, tck)
        spec_cuml_new[log_w_new >= log_w_old[-1]] = log_w_old[-1]
        spec_cuml_new[log_w_new <= log_w_old[0]] = 0
        snew.spectrum = np.diff(spec_cuml_new)
        snew.spectrum *= self.coeff1 / snew.coeff1

        # interpolate error
        err_cuml_old = self.error.cumsum()
        tck = interpolate.splrep(log_w_old, np.hstack(([0], err_cuml_old)))
        err_cuml_new = interpolate.splev(log_w_new, tck)
        err_cuml_new[log_w_new >= log_w_old[-1]] = log_w_old[-1]
        err_cuml_new[log_w_new <= log_w_old[0]] = 0
        snew.error = np.diff(err_cuml_new)
        snew.error *= self.coeff1 / snew.coeff1

        return snew

    def _get_line_strength(self, line):
        lam = LINES.get(line)
        if lam is None:
            lam1 = LINES.get(line + 'a')
            ind1 = np.where(abs(self.hdulist[2].data['restWave']
                                - lam1) < 1)[0]

            lam2 = LINES.get(line + 'b')
            ind2 = np.where(abs(self.hdulist[2].data['restWave']
                                - lam2) < 1)[0]

            if len(ind1) == 0:
                s1 = h1 = 0
                nsig1 = 0
            else:
                s1 = self.hdulist[2].data['sigma'][ind1]
                h1 = self.hdulist[2].data['height'][ind1]
                nsig1 = self.hdulist[2].data['nsigma'][ind1]

            if len(ind2) == 0:
                s2 = h2 = 0
                nsig2 = 0
            else:
                s2 = self.hdulist[2].data['sigma'][ind2]
                h2 = self.hdulist[2].data['height'][ind2]
                nsig2 = self.hdulist[2].data['nsigma'][ind2]

            strength = s1 * h1 + s2 * h2
            nsig = max(nsig1, nsig2)
        else:
            ind = np.where(abs(self.hdulist[2].data['restWave'] - lam) < 1)[0]

            if len(ind) == 0:
                strength = 0
                nsig = 0
            else:
                s = self.hdulist[2].data['sigma'][ind]
                h = self.hdulist[2].data['height'][ind]
                nsig = self.hdulist[2].data['nsigma'][ind]
                strength = s * h

        return strength, nsig

    def lineratio_index(self, indicator='NII'):
        """Return the line ratio index for the given galaxy.

        This is the index used in Vanderplas et al 2009, and makes use
        of line-ratio fits from Kewley et al 2001

        Parameters
        ----------
        indicator: string ['NII'|'OI'|'SII']
            The emission line to use as an indicator

        Returns
        -------
        cln: integer
            The classification of the spectrum based on SDSS pipeline and
            the line ratios.

            0 : unknown (SPEC_CLN = 0)
            1 : star (SPEC_CLN = 1)
            2 : absorption galaxy (H-alpha seen in absorption)
            3 : normal galaxy (no significant H-alpha emission or absorption)
            4 : emission line galaxies (below line-ratio curve)
            5 : narrow-line QSO (above line-ratio curve)
            6 : broad-line QSO (SPEC_CLN = 3)
            7 : Sky (SPEC_CLN = 4)
            8 : Hi-z QSO (SPEC_CLN = 5)
            9 : Late-type star (SPEC_CLN = 6)
            10 : Emission galaxy (SPEC_CLN = 7)

        ratios: tuple
            The line ratios used to compute this
        """
        assert indicator in ['NII', 'OI', 'SII']

        if self.spec_cln < 2:
            return self.spec_cln, (0, 0)
        elif self.spec_cln > 2:
            return self.spec_cln + 3, (0, 0)

        strength_Ha, nsig_Ha = self._get_line_strength('Ha')
        strength_Hb, nsig_Hb = self._get_line_strength('Hb')
        if nsig_Ha < 3 or nsig_Hb < 3:
            return 3, (0, 0)

        if strength_Ha < 0 or strength_Hb < 0:
            return 2, (0, 0)

        # all that's left is choosing between 4 and 5
        # we do this based on line-ratios
        strength_I, nsig_I = self._get_line_strength(indicator)
        strength_OIII, nsig_OIII = self._get_line_strength('OIII')

        log_OIII_Hb = np.log10(strength_OIII / strength_Hb)
        I_Ha = np.log10(strength_I / strength_Ha)

        if indicator == 'NII':
            if I_Ha >= 0.47 or log_OIII_Hb >= log_OIII_Hb_NII(I_Ha):
                return 5, (I_Ha, log_OIII_Hb)
            else:
                return 4, (I_Ha, log_OIII_Hb)

        elif indicator == 'OI':
            if I_Ha >= -0.59 or log_OIII_Hb >= log_OIII_Hb_OI(I_Ha):
                return 5, (I_Ha, log_OIII_Hb)
            else:
                return 4, (I_Ha, log_OIII_Hb)

        else:
            if I_Ha >= 0.32 or log_OIII_Hb >= log_OIII_Hb_SII(I_Ha):
                return 5, (I_Ha, log_OIII_Hb)
            else:
                return 4, (I_Ha, log_OIII_Hb)


#----------------------------------------------------------------------
# Empirical fits from Kewley et al 2001
def log_OIII_Hb_NII(log_NII_Ha, eps=0):
    return 1.19 + eps + 0.61 / (log_NII_Ha - eps - 0.47)


def log_OIII_Hb_OI(log_OI_Ha, eps=0):
    return 1.33 + eps + 0.73 / (log_OI_Ha - eps + 0.59)


def log_OIII_Hb_SII(log_SII_Ha, eps=0):
    return 1.30 + eps + 0.72 / (log_SII_Ha - eps - 0.32)
