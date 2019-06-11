import numpy as np

from . import sql_query

# SDSS primtarget codes
TARGET_QSO_HIZ            = int('0x00000001', 16)
TARGET_QSO_CAP	          = int('0x00000002', 16)
TARGET_QSO_SKIRT	  = int('0x00000004', 16)
TARGET_QSO_FIRST_CAP	  = int('0x00000008', 16)
TARGET_QSO_FIRST_SKIRT    = int('0x00000010', 16)
TARGET_GALAXY_RED	  = int('0x00000020', 16)
TARGET_GALAXY	          = int('0x00000040', 16)
TARGET_GALAXY_BIG	  = int('0x00000080', 16)
TARGET_GALAXY_BRIGHT_CORE = int('0x00000100', 16)
TARGET_ROSAT_A            = int('0x00000200', 16)
TARGET_ROSAT_B            = int('0x00000400', 16)
TARGET_ROSAT_C            = int('0x00000800', 16)
TARGET_ROSAT_D            = int('0x00001000', 16)
TARGET_STAR_BHB           = int('0x00002000', 16)
TARGET_STAR_CARBON        = int('0x00004000', 16)
TARGET_STAR_BROWN_DWARF   = int('0x00008000', 16)
TARGET_STAR_SUB_DWARF     = int('0x00010000', 16)
TARGET_STAR_CATY_VAR      = int('0x00020000', 16)
TARGET_STAR_RED_DWARF     = int('0x00040000', 16)
TARGET_STAR_WHITE_DWARF   = int('0x00080000', 16)
TARGET_SERENDIP_BLUE      = int('0x00100000', 16)
TARGET_SERENDIP_FIRST     = int('0x00200000', 16)
TARGET_SERENDIP_RED       = int('0x00400000', 16)
TARGET_SERENDIP_DISTANT   = int('0x00800000', 16)
TARGET_SERENDIP_MANUAL    = int('0x01000000', 16)
TARGET_QSO_FAINT          = int('0x02000000', 16)
TARGET_GALAXY_RED_II      = int('0x04000000', 16)
TARGET_ROSAT_E            = int('0x08000000', 16)
TARGET_STAR_PN            = int('0x10000000', 16)
TARGET_QSO_REJECT         = int('0x20000000', 16)

DEFAULT_TARGET = TARGET_GALAXY  # main galaxy sample


def query_plate_mjd_fiber(n_spectra,
                          primtarget=DEFAULT_TARGET,
                          zmin=0, zmax=0.7):
    """Query the SDSS server for plate, mjd, and fiber numbers

    Parameters
    ----------
    n_spectra: int
        number of spectra to query.  Max is 100,000 (set by CAS server)
    primtarget: int
        prime target flag.  See notes below
    zmin, zmax: float
        minimum and maximum redshift range for query

    Returns
    -------
    plate, mjd, fiber : ndarrays, size=n_spectra
        The plate numbers MJD, and fiber numbers of the spectra

    Notes
    -----
    Primtarget flag values can be found at
    http://cas.sdss.org/dr7/en/help/browser/enum.asp?n=PrimTarget
    """
    query_text = '\n'.join(("SELECT TOP %(n_spectra)i ",
                            "  plate, mjd, fiberid ",
                            "FROM specObj ",
                            "WHERE ((PrimTarget & %(primtarget)i) > 0) ",
                            "       AND (z > %(zmin)f)",
                            "       AND (z <= %(zmax)f) ")) % locals()

    output = sql_query(query_text).readlines()
    keys = output[0]

    res = np.zeros((n_spectra, 3), dtype=int)
    for i, line in enumerate(output[2:]):
        try:
            res[i] = line.decode().strip().split(',')
        except BaseException:
            raise ValueError(b'\n'.join(output))

    ntot = i + 1

    return res[:ntot].T
