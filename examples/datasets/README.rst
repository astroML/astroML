Data set Examples
-----------------
These plots show some of the data set loaders available in astroML, and some
of the ways that astronomical data can be visualized and processed using
open source python tools.  The dataset loaders are in the submodule
:mod:`astroML.datasets`, and start with the word ``fetch_``.

The first time a dataset loader is called, it will attempt to download the
dataset from the web and store it locally on disk.  The default location
is ``~/astroML_data``, but this location can be changed by specifying an
alternative directory in the ``ASTROML_DATA`` environment variable.  On
subsequent calls, the cached version of the data is used.

For more examples, see the :ref:`figures <book_fig_root>` from the textbook.
