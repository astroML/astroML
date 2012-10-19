"""
Generate examples based on the files in Machine_Learning_Code/book_figures/

We'll first use the 'contents.txt' file to find what subdirectories should
be used, and in what order.
Within these directories, we'll again use the file 'contents.txt' to
execute the figure code in the correct order.

We need to generate the following:

 auto_book_figures/index.rst
   This should include a toctree with references to each of the main
   chapter indices
 
 auto_book_figures/chapter*/index.rst
   This should include a toctree with references to fig*.rst within
   each chapter folder, with a one-line description of each

 auto_book_figures/chapter*/fig_*.rst
   Individual example files.  These should have a link to the source-code,
   a (hideable?) in-page version of the source code, the output included,
   and thumbnails of the images that link to the image pages

 auto_book_figures/chapter*/fig_*-*.rst
   Image pages, which display a single image, along with explanatory captions
   for the automatically generated images

 auto_text_exammples/chapter*/fig_*.out
   stdout output of the example.  This will be included in the example page

 auto_text_exammples/chapter*/fig_*.py
   copy of example source code.  This will be included in the example page,
   minus the doc string.

 auto_book_figures/images/chapter*/images/fig*-*.png
 auto_book_figures/images/chapter*/thumbnails/fig*-*.png
   Images and their thumbnails.
"""
import sys
import os
import shutil
import traceback
import glob
import token, tokenize

import matplotlib
matplotlib.use('Agg') #don't display plots

from matplotlib import image

from exfile import ExecFile

# HACK: for some reason, importing scipy here fixes some errors.
# if scipy is imported in the execfile statements below and not here,
# then the build breaks.  Weird.
import scipy

IMAGE_FORMAT_HEADER =\
"""

.. raw:: html

    <style type="text/css">
    .figure {
        float: left;
        margin: 10px;
        width: auto;
        height: 200px;
        width: 180px;
    }

    .figure img {
        display: inline;
        }

    .figure .caption {
        width: 170px;
        text-align: center !important;
    }
    </style>

"""

PYTHON_INCLUDE =\
"""

.. raw:: html
    
    <div class="toggle_trigger"><a href="#">


**Python source code:**

.. raw:: html

    </a></div>
    <div class="toggle_container">

.. literalinclude:: %(fname)s
    :lines: %(end_line)s-

.. raw:: html
    
    </div>
    <div align="right">

[:download:`%(fname)s <%(fname)s>`]

.. raw:: html
    
    </div>

"""

OUTPUT_INCLUDE =\
"""

.. raw:: html
    
    <div class="toggle_trigger"><a href="#">

**Python code output:**

.. raw:: html

    </a></div>
    <div class="toggle_container">

.. literalinclude:: %(fname)s

.. raw:: html
    
    </div>
    <div align="right">

[:download:`%(fname)s <%(fname)s>`]

.. raw:: html
    
    </div>

"""

IMLIST_HEADER = """

.. raw:: html
    
    <div class="toggle_trigger" id="start_open"><a href="#">

**Figures:**

.. raw:: html

    </a></div>
    <div class="toggle_container">
    <div class="block">

"""

IMLIST_IMAGE_TEMPLATE = """

.. image:: thumbnails/%(fname)s
   :target: %(html_fname)s

"""

SINGLE_IMAGE_TEMPLATE = """

.. image:: images/%(fname)s

"""

SINGLE_IMAGE_TEMPLATE_W100 = """

.. image:: images/%(fname)s
   :width: 100%%

"""

IMLIST_FOOTER = """

.. raw:: html
    
    </div></div>
"""

def create_thumbnail(infile, thumbfile, scale=0.2,
                     max_height=150, interpolation='bilinear'):
    """
    Create a thumbnail of an image
    """
    basedir, basename = os.path.split(infile)
    baseout, extout = os.path.splitext(thumbfile)

    im = image.imread(infile)
    rows, cols, depth = im.shape

    # this doesn't really matter, it will cancel in the end, but we
    # need it for the mpl API
    dpi = 100

    scale = min(scale, max_height / float(rows))

    height = float(rows) / dpi * scale
    width = float(cols) / dpi * scale

    extension = extout.lower()

    if extension=='.png':
        from matplotlib.backends.backend_agg \
            import FigureCanvasAgg as FigureCanvas
    elif extension=='.pdf':
        from matplotlib.backends.backend_pdf \
            import FigureCanvasPDF as FigureCanvas
    elif extension=='.svg':
        from matplotlib.backends.backend_svg \
            import FigureCanvasSVG as FigureCanvas
    else:
        raise ValueError("Can only handle extensions 'png', 'svg' or 'pdf'")

    from matplotlib.figure import Figure
    fig = Figure(figsize=(width, height), dpi=dpi)
    canvas = FigureCanvas(fig)

    ax = fig.add_axes([0,0,1,1], aspect='auto',
                      frameon=False, xticks=[], yticks=[])

    basename, ext = os.path.splitext(basename)
    ax.imshow(im, aspect='auto', resample=True,
              interpolation='bilinear')
    fig.savefig(thumbfile, dpi=dpi)
    return fig



def read_contents(contents_file):
    """Read contents file

    A contents file is a list of filenames within the directory,
    with comments marked by '#' in the normal way.

    return a list of filenames.
    
    raise a ValueError if a file listed in the contents doesn't exist.
    """
    contents_file = os.path.abspath(contents_file)
    directory = os.path.dirname(contents_file)

    if not os.path.exists(contents_file):
        return []

    files = os.listdir(directory)
    L = []
    for line in open(contents_file):
        filename = line.split('#')[0].strip()
        if len(filename) == 0:
            continue

        if filename not in files:
            raise ValueError("Fatal: file %s not found in %s" % (filename,
                                                                 directory))

        if not filename.endswith('.py'):
            raise ValueError("Fatal: file %s is not a python script" % filename)
        
        L.append(filename)

    return L


def generate_figures_rst(app):
    """ Generate the list of examples, contents, and output

    This is the master function: it will call generate_chapter_rst
    for each of the chapter subdirectories.
    """
    root_dir = os.path.join(app.builder.srcdir, 'auto_book_figures')
    code_dir = os.path.abspath(os.path.join(app.builder.srcdir,
                                               '../', 'book_figures'))

    # get the keyword from the build
    try:
        figure_gallery = eval(app.builder.config.figure_gallery)
    except TypeError:
        figure_gallery = bool(app.builder.config.figure_gallery)

    # create directories if needed
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if ncot os.path.exists(code_dir):
        os.makedirs(code_dir)

    root_index = file(os.path.join(root_dir, 'index.rst'), 'w')
    root_index.write("\n\n"
                    ".. _text-figures-index:\n\n"
                    ".. toctree::\n"
                    "   :numbered:\n"
                    "   :maxdepth: 2\n")
    
    chapter_list = open(os.path.join(code_dir, 'contents.txt')).read().split()

    for chapter in chapter_list:
        generate_chapter_rst(chapter, root_index, code_dir,
                             root_dir, figure_gallery)
    root_index.flush()


def generate_chapter_rst(chapter, root_index, code_dir,
                         root_dir, figure_gallery):
    """Generate rst for a chapter

    This will call generate_single_figure for each figure in contents.txt
    """
    source_chapter_dir = os.path.join(code_dir, chapter)
    dest_chapter_dir = os.path.join(root_dir, chapter)

    if not os.path.exists(dest_chapter_dir):
        os.makedirs(dest_chapter_dir)
        
    chapter_num = int(chapter[7:]) # remove 'chapter' from front
    
    chapter_index = open(os.path.join(dest_chapter_dir, 'index.rst'), 'w')
                          
    contents_file = os.path.join(source_chapter_dir, 'contents.txt')
    description_file = os.path.join(source_chapter_dir, 'header.rst')

    figure_file_list = read_contents(contents_file)

    if not os.path.exists(description_file):
        chapter_index.write("\n"
                            "Chapter %i\n"
                            "============\n\n" % chapter_num)

    else:
        chapter_index.write("\n\n%s\n\n" % open(description_file).read())

    chapter_index.write(".. _examples-chapter%i-index:\n\n"
                        ".. toctree::\n"
                        "   :maxdepth: 1\n"
                        "   :titlesonly:\n\n" % (chapter_num))
                        
    root_index.write("\n   %s/index\n" % chapter)

    if os.path.exists(contents_file):
        contents_modtime = os.stat(contents_file).st_mtime
    else:
        contents_modtime = os.stat(root_index.name).st_mtime

    for i, figure_file in enumerate(figure_file_list):
        figure_num = i + 1
        generate_single_figure(figure_file, chapter_num, figure_num,
                               contents_modtime,
                               chapter_index, source_chapter_dir,
                               dest_chapter_dir, figure_gallery)


def generate_single_figure(figure_file, chapter_num, figure_num,
                           contents_modtime,
                           chapter_index, source_chapter_dir,
                           dest_chapter_dir, figure_gallery):
    figure_basename = "figure%i.%i" % (chapter_num, figure_num)

    # python source and destination
    figure_src_file = os.path.join(source_chapter_dir, figure_file)
    figure_py_file = os.path.join(dest_chapter_dir, figure_file)
    
    chapter_index.write('   %s\n\n' % figure_basename)
    
    # create an rst file for this figure
    figure_rst_file = open(os.path.join(dest_chapter_dir,
                                         figure_basename + '.rst'), 'w')
    
    figure_rst_file.write(IMAGE_FORMAT_HEADER)

    # set up the image directories
    image_dir = os.path.join(dest_chapter_dir, 'images')
    thumb_dir = os.path.join(dest_chapter_dir, 'thumbnails')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(thumb_dir):
        os.makedirs(thumb_dir)
    
    # save images to include in page
    image_filename = figure_basename + '_%s.png'
    image_path = os.path.join(image_dir,
                              image_filename)
    thumb_path = os.path.join(thumb_dir,
                              image_filename)
    output_file = figure_py_file.rstrip('.py') + '.out'

    #execute only if output is older than the script
    if os.path.exists(output_file):
        output_modtime = os.stat(output_file).st_mtime
    else:
        output_modtime = -9999
    source_modtime = os.stat(figure_src_file).st_mtime

    if (not os.path.exists(output_file) or
        (output_modtime <= source_modtime) or
        (output_modtime <= contents_modtime)):
        EF = ExecFile(figure_src_file, execute=figure_gallery)
        print " >", figure_basename
        
        # write output file
        open(output_file, 'w').write(EF.output)

        # get list of figures
        figure_list = []
        for fig in EF.figlist:
            fig.savefig(image_path % fig.number)
            create_thumbnail(image_path % fig.number,
                             thumb_path % fig.number,
                             0.2, 150)
            figure_list.append(image_filename % fig.number)

        # copy source code file and include in page
        shutil.copyfile(figure_src_file, figure_py_file)

    else:
        EF = ExecFile(figure_src_file, execute=False)
        figure_list = [os.path.basename(f)
                       for f in glob.glob(image_path % '[1-9]')]

            
        
    # write the doc string to the rst file
    figure_rst_file.write("\n\n%s\n\n" % EF.docstring)

    figure_rst_file.write(PYTHON_INCLUDE % dict(fname=figure_file,
                                                end_line=EF.end_line))

    if os.path.exists(output_file) and open(output_file).read() != '':
        figure_rst_file.write(OUTPUT_INCLUDE % 
                               dict(fname=os.path.basename(output_file)))

    if len(figure_list) > 0:
        figure_rst_file.write(IMLIST_HEADER)

        for figfile in figure_list:
            figure_rst_file.write(IMLIST_IMAGE_TEMPLATE
                                   % dict(fname=figfile,
                                          html_fname=(figfile.rstrip('.png')
                                                      + '.html')))
            

        if len(EF.figlist) > 0:
            for fig, figfile in zip(EF.figlist, figure_list):
                #create an image file
                html_file = figfile.rstrip('.png') + '.rst'
                html_F = open(os.path.join(dest_chapter_dir,
                                           html_file),'w')
                
                html_F.write("========================\n"
                             "Figure %i.%i, panel %i\n"
                             "========================\n\n"
                             % (chapter_num, figure_num, fig.number))
                if hasattr(fig, 'caption'):
                    html_F.write("\n\n%s\n\n" % fig.caption)
                    
                if fig.get_figwidth() > 8:
                    html_F.write(SINGLE_IMAGE_TEMPLATE_W100 
                                 % dict(fname=figfile))
                else:
                    html_F.write(SINGLE_IMAGE_TEMPLATE % dict(fname=figfile))

        figure_rst_file.write(IMLIST_FOOTER)

        #figure_rst_file.write(".. toctree::\n"
        #                       "   :hidden:\n\n")    
        #for figfile in figure_list:
        #    figure_rst_file.write('   %s\n' % figfile[:-4])


def setup(app):
    app.connect('builder-inited', generate_figures_rst)
    app.add_config_value('figure_gallery', True, 'html')
