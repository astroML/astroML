"""
Astronomy Datasets
------------------
"""

from .tools import get_data_home
from .sdss_S82standards import fetch_sdss_S82standards
from .dr7_quasar import fetch_dr7_quasar
from .moving_objects import fetch_moving_objects
from .sdss_galaxy_colors import fetch_sdss_galaxy_colors
from .sdss_spectrum import fetch_sdss_spectrum
from .sdss_corrected_spectra import fetch_sdss_corrected_spectra
from .nasa_atlas import fetch_nasa_atlas
from .sdss_sspp import fetch_sdss_sspp
from .sdss_specgals import fetch_sdss_specgals, fetch_great_wall
from .imaging_sample import fetch_imaging_sample
from .wmap_temperatures import fetch_wmap_temperatures
from .rrlyrae_mags import fetch_rrlyrae_mags, fetch_rrlyrae_combined
from .LINEAR_sample import fetch_LINEAR_sample, fetch_LINEAR_geneva
from .LIGO_bigdog import fetch_LIGO_bigdog, fetch_LIGO_large
from .generated import generate_mu_z
from .hogg2010test import fetch_hogg2010test
from .rrlyrae_templates import fetch_rrlyrae_templates
from .sdss_filters import fetch_sdss_filter, fetch_vega_spectrum
from .kelly2007test import simulation_kelly
