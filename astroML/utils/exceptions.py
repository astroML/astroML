"""
This module contains errors/exceptions and warnings for astroML.
"""
from astropy.utils.exceptions import AstropyWarning


class AstroMLWarning(AstropyWarning):
    """
    A base warning class from which all AstroML warnings should inherit.

    This class is subclassed from AstropyWarnings, so warnings inherited by
    this class is handled by the Astropy logger.

    """


class AstroMLDeprecationWarning(AstroMLWarning):
    """
    A warning class to indicate a deprecated feature in astroML.

    """
