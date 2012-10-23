=======================================================
AstroML: Machine Learning and Data Mining for Astronomy
=======================================================
..
   Here we are building a banner: a javascript selects randomly 4 images in
   the list.  Code adapted from scikit-learn documentation

.. only:: html

    .. |banner1| image:: examples/images/datasets/plot_moving_objects_2.png
       :height: 150
       :target: examples/datasets/plot_moving_objects.html

    .. |banner2| image:: examples/images/datasets/plot_wmap_power_spectra_1.png
       :height: 150
       :target: examples/datasets/plot_wmap_power_spectra.html

    .. |banner3| image:: book_figures/images/chapter3/fig_uniform_mean_1.png
       :height: 150
       :target: book_figures/chapter3/fig_uniform_mean.html

    .. |banner4| image:: book_figures/images/chapter4/fig_lyndenbell_gals_1.png
       :height: 150
       :target: book_figures/chapter4/fig_lyndenbell_gals.html

    .. |banner5| image:: examples/images/datasets/plot_moving_objects_1.png
       :height: 150
       :target: examples/datasets/plot_moving_objects.html

    .. |banner6| image:: book_figures/images/chapter7/fig_spec_LLE_1.png
       :height: 150
       :target: book_figures/chapter7/fig_spec_LLE.html

    .. |banner7| image:: examples/images/datasets/plot_nasa_atlas_1.png
       :height: 150
       :target: examples/datasets/plot_nasa_atlas.html

    .. |banner8| image:: examples/images/datasets/plot_sdss_line_ratios_1.png
       :height: 150
       :target: examples/datasets/plot_sdss_line_ratios.html

    .. |banner9| image:: book_figures/images/chapter10/fig_LINEAR_clustering_1.png
       :height: 150
       :target: book_figures/chapter10/fig_LINEAR_clustering.html

    .. |banner10| image:: book_figures/images/appendix/fig_sdss_filters_1.png
       :height: 150
       :target: book_figures/appendix/fig_sdss_filters.html

    .. |center-div| raw:: html

        <div style="text-align: center; margin: -7px 0 -10px 0;" id="banner">

    .. |end-div| raw:: html

        </div>

        <SCRIPT>
        // Function to select 4 imgs in random order from a div
        function shuffle(e) {       // pass the divs to the function
          var replace = $('<div>');
          var size = 4;
          var num_choices = e.size();

          while (size >= 1) {
            var rand = Math.floor(Math.random() * num_choices);
            var temp = e.get(rand);      // grab a random div from our set
            replace.append(temp);        // add the selected div to our new set
            e = e.not(temp); // remove our selected div from the main set
            size--;
            num_choices--;
          }
          $('#banner').html(replace.html() ); // update our container div
                                              // with the new, randomized divs
        }
        shuffle ($('#banner a.external'));

        </SCRIPT>

    |center-div| |banner1| |banner2| |banner3| |banner4| |banner5| |banner6| |banner7| |banner8| |banner9| |banner10| |end-div|

.. only:: html

 .. sidebar:: Download 
    
    * Source code: `github <https://github.com/astroML/astroML>`_

    * Source tarball: :download:`astroML_0.1.tgz <_static/astroML_0.1.tgz>`

.. sectionauthor:: Jake Vanderplas <vanderplas@astro.washington.edu>

AstroML is a Python module for machine learning and data mining built on
`numpy <http://numpy.scipy.org>`_,
`scipy <http://scipy.org>`_,
`scikit-learn <http://scikit-learn.org>`_,
and `matplotlib <http://matplotlib.org>`_,
and distributed under the 3-clause BSD license.
It contains a growing library of statistical and machine learning
routines for analyzing astronomical data in python, loaders for several open
astronomical datasets, and a large suite of examples of analyzing and
visualizing astronomical datasets.

.. image:: _static/text_cover.png
     :width: 100 px
     :align: left

The goal of astroML is to provide a community repository for fast Python
implementations of common tools and routines used for statistical
data analysis in astronomy and astrophysics, to provide a uniform and
easy-to-use interface to freely available astronomical datasets.
We hope this package will be useful to researchers and students of
astronomy.  The project was started in 2012 to accompany the book
**Statistics, Data Mining, and Machine Learning in Astronomy** by
Zeljko Ivezic, Andrew Connolly, Jacob VanderPlas, and Alex Gray,
to be published in early 2013.

.. include:: includes/big_toc_css.rst

User Guide
==========

.. toctree::
   :maxdepth: 2

   user_guide/index

Example Plot Galleries
======================

.. toctree::
   :maxdepth: 2

   examples/index
   book_figures/index
   paper_figures/index

Development
===========

.. toctree::
   :maxdepth: 2

   development/index

