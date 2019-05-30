import os

from .tools import get_data_home, download_with_progress_bar,\
    SDSSfits, sdss_fits_url, sdss_fits_filename


def fetch_sdss_spectrum(plate, mjd, fiber, data_home=None,
                        download_if_missing=True,
                        cache_to_disk=True):
    """Fetch an SDSS spectrum from the Data Archive Server

    Parameters
    ----------
    plate: integer
        plate number of desired spectrum
    mjd: integer
        mean julian date of desired spectrum
    fiber: integer
        fiber number of desired spectrum

    Other Parameters
    ----------------
    data_home: string (optional)
        directory in which to cache downloaded fits files.  If not
        specified, it will be set to ~/astroML_data.
    download_if_missing: boolean (default = True)
        download the fits file if it is not cached locally.
    cache_to_disk: boolean (default = True)
        cache downloaded file to data_home.

    Returns
    -------
    spec: :class:`astroML.tools.SDSSfits` object
        An object wrapper for the fits data
    """
    data_home = get_data_home(data_home)

    target_url = sdss_fits_url(plate, mjd, fiber)
    target_file = os.path.join(data_home, 'SDSSspec', '%04i' % plate,
                               sdss_fits_filename(plate, mjd, fiber))

    if not os.path.exists(target_file):
        if not download_if_missing:
            raise IOError("SDSS colors training data not found")

        buf = download_with_progress_bar(target_url, return_buffer=True)

        if cache_to_disk:
            print("caching to %s" % target_file)
            if not os.path.exists(os.path.dirname(target_file)):
                os.makedirs(os.path.dirname(target_file))
            fhandler = open(target_file, 'wb')
            fhandler.write(buf.read())
            buf.seek(0)
    else:
        buf = target_file

    return SDSSfits(buf)
