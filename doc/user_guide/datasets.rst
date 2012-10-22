.. _datasets_description:

Data Sets
=========
One of the major components of astroML is its tools for downloading and
working with astronomical data sets.  The available routines are available
in the module :mod:`astroML.datasets`, and details are available in the
documentation of the functions therein.  In this section we will summarize
some of the data sets made available by the code, and show some visualizations
of this data

.. currentmodule:: astroML.datasets

Data Set Cache Location
-----------------------
The total size of the data sets is in the hundreds of megabytes, too large to
be bundled with the astroML source distribution.  To make working with
data sets more convenient, astroML contains routines which download the
data from their locations on the web, and cache the results to disk for
future use.

For example, the :func:`fetch_sdss_spectrum` function queries the SDSS
database using an SQL query, and retrieves the data.  The file is stored
to disk in a location that can be defined by the user.  The default location
is ``~/astroML_data``, and this default can be overridden by setting the
``ASTROML_DATA`` environment variable.  Any subsequent time the same function
is called, the cached version of the data is used automatically.

SDSS Data
---------
Much of the data made available by astroML comes from the
`Sloan Digital Sky Survey (SDSS) <http://www.sdss.org>`_, a decade-plus
photometric and spectroscopic survey at the Apache Point Observatory in
New Mexico.  The survey obtained photometry for hundreds of millions of
stars, quasars, and galaxies, and spectra for several million of these
objects.  In addition, the second phase of the survey performed repeated
imaging over a small portion of the sky, called Stripe 82, enabling the
study of the time-variation of many objects.

SDSS Spectra
~~~~~~~~~~~~
The SDSS spectroscopic data is available in a database, indexed by three
numbers: the plate, date, and fiber number.  The :ref:`fetch_sdss_spectrum`
takes a plate, mjd, and fiber, and downloads the spectrum to disk.  The
spectral data can be visualized as follows:

.. image:: ../examples/images/datasets/plot_sdss_spectrum_1.png
   :target: ../examples/datasets/plot_sdss_spectrum.html
   :align: center
   :scale: 80

As with all figures in this documentation, clicking on the image will link
to a page showing the source code used to download the data and plot the
result.





