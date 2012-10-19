============================================
AstroML: Machine Learning code for Astronomy
============================================
..
   Here we are building a banner: a javascript selects randomly 4 images in
   the list.  Code adapted from scikit-learn documentation

.. only:: html

    .. |banner1| image:: auto_book_figures/images/chapter1/fig_moving_objects_multicolor_1.png
       :height: 130
       :target: auto_book_figures/chapter1/fig_moving_objects_multicolor.html

    .. |banner2| image:: auto_book_figures/images/chapter1/fig_wmap_healpix_1.png
       :height: 130
       :target: auto_book_figures/chapter1/fig_wmap_healpix.html

    .. |banner3| image:: auto_book_figures/images/chapter3/fig_uniform_mean_1.png
       :height: 130
       :target: auto_book_figures/chapter3/fig_uniform_mean.html

    .. |banner4| image:: auto_book_figures/images/chapter4/fig_lyndenbell_gals_1.png
       :height: 130
       :target: auto_book_figures/chapter4/fig_lyndenbell_gals.html

    .. |banner5| image:: auto_book_figures/images/chapter6/fig_great_wall_MST_1.png
       :height: 130
       :target: auto_book_figures/chapter3/fig_great_wall_MST.html

    .. |banner6| image:: auto_book_figures/images/chapter7/fig_spec_LLE_1.png
       :height: 130
       :target: auto_book_figures/chapter7/fig_spec_LLE.html

    .. |banner7| image:: auto_book_figures/images/chapter8/fig_nonlinear_mu_z_1.png
       :height: 130
       :target: auto_book_figures/chapter8/fig_nonlinear_mu_z.html

    .. |banner8| image:: auto_book_figures/images/chapter10/fig_wavelet_PSD_1.png
       :height: 130
       :target: auto_book_figures/chapter10/fig_wavelet_PSD.html

    .. |banner9| image:: auto_book_figures/images/chapter10/fig_LINEAR_clustering_1.png
       :height: 130
       :target: auto_book_figures/chapter10/fig_LINEAR_clustering.html

    .. |banner10| image:: auto_book_figures/images/appendix/fig_sdss_filters_1.png
       :height: 130
       :target: auto_book_figures/appendix/fig_sdss_filters.html

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

	function preload_images() {
	    var img = new Image();
	    img.src="_static/plusBox.png";
	    img.src="_static/minBox.png";
	    img.src="_static/plusBoxHighlight.png";
	    img.src="_static/minBoxHighlight.png";
	    img.src="_static/noneBox.png";
	}
	preload_images();

	//Function to make the index toctree collapsible
	$(function () {
            $('.toctree-l2')
                .click(function(event){
                    if (event.target.tagName.toLowerCase() != "a") {
		        if ($(this).children('ul').length > 0) {
                            $(this).css('list-style-image',
                            (!$(this).children('ul').is(':hidden')) ? 'url(_static/plusBoxHighlight.png)' : 'url(_static/minBoxHighlight.png)');
                            $(this).children('ul').toggle();
                        }
                        return true; //Makes links clickable
                    }
		})
		.mousedown(function(event){ return false; }) //Firefox highlighting fix
                .css({cursor:'pointer', 'list-style-image':'url(_static/plusBox.png)'})
                .children('ul').hide();
            $('ul li ul li:not(:has(ul))').css({cursor:'default', 'list-style-image':'url(_static/noneBox.png)'});
	    $('.toctree-l3').css({cursor:'default', 'list-style-image':'url(_static/noneBox.png)'});
            var sidebarbutton = $('#sidebarbutton');
            sidebarbutton.css({
	        'display': 'none'
            });

	    $('.toctree-l2').hover(
	        function () {
		    if ($(this).children('ul').length > 0) {
		        $(this).css('background-color', '#D0D0D0').children('ul').css('background-color', '#F0F0F0');
		        $(this).css('list-style-image',
                            (!$(this).children('ul').is(':hidden')) ? 'url(_static/minBoxHighlight.png)' : 'url(_static/plusBoxHighlight.png)');
		    }
		    else {
		        $(this).css('background-color', '#F9F9F9');
		    }
                },
                function () {
                    $(this).css('background-color', 'white').children('ul').css('background-color', 'white');
		    if ($(this).children('ul').length > 0) {
		        $(this).css('list-style-image',
                            (!$(this).children('ul').is(':hidden')) ? 'url(_static/minBox.png)' : 'url(_static/plusBox.png)');
		    }
                }
            );
	});

        </SCRIPT>

    |center-div| |banner1| |banner2| |banner3| |banner4| |banner5| |banner6| |banner7| |banner8| |banner9| |banner10| |end-div|

AstroML is a Python module for machine learning and data mining
built on numpy, scipy, scikit-learn, and matplotlib,
and distributed under the 3-Clause BSD license.
It contains a growing library of statistical and machine learning
routines for analyzing astronomical data in python, loaders for several open
astronomical datasets, and a large suite of examples of analyzing and
visualizing astronomical datasets.

This project was started in 2012 by Jake VanderPlas to accompany the book
*Statistics, Data Mining, and Machine Learning in Astronomy* by
Zeljko Ivezic, Andrew Connolly, Jacob VanderPlas, and Alex Gray.

The project is split into two components.  The core ``astroML`` library is
written in python only, and is designed to be very easy to install for
any users, even those who don't have a working C or fortran compiler.
A companion library, ``astroML_addons``, can be optionally installed for
increased performance on certain algorithms.  Every algorithm
in ``astroML_addons`` exists in the core ``astroML`` implementation, but the
``astroML_addons`` library contains faster and more efficient implementations.
Furthermore, if ``astroML_addons`` is installed on your system, the core
``astroML`` library will import and use the faster routines by default.

Example Plot Galleries
======================

.. toctree::
   :maxdepth: 2

   auto_examples/index
   auto_book_figures/index
   auto_paper_figures/index

Information
===========

.. toctree::
   :maxdepth: 2

   installation
   development

Appendix
========

.. toctree::
   :maxdepth: 2

   contents

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Authors
=======
Jake Vanderplas <vanderplas@astro.washington.edu> http://jakevdp.github.com
