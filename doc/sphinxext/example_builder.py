"""
Example Builder:
a tool for creating automated example pages in sphinx documentation
"""

import sys
import os
import traceback
import token
import tokenize
import gc
import shutil
import glob
import warnings

import matplotlib
matplotlib.use('Agg')  # don't display plots

# set some font properties
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif', style='normal', variant='normal',
              stretch='normal', weight='normal',)

from matplotlib import image
from matplotlib import pyplot as plt

DEFAULT_RST_TEMPLATE = """

.. _%(sphinx_tag)s:

%(docstring)s

%(image_list)s

**Code output:**

.. literalinclude:: %(stdout)s

**Python source code:** :download:`[download source: %(fname)s] <%(fname)s>`

.. literalinclude:: %(fname)s
    :lines: %(end_line)s-

"""

DEFAULT_INDEX_TEMPLATE = """

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

.. _%(sphinx_tag)s:

%(info)s

%(subdir_contents)s

%(figure_contents)s

%(footer)s

.. raw:: html

    <div style="clear: both"></div>

"""


def create_thumbnail(infile, thumbfile, scale=0.2,
                     max_width=200, max_height=150,
                     interpolation='bilinear'):
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
    scale = min(scale, max_width / float(cols))

    height = float(rows) / dpi * scale
    width = float(cols) / dpi * scale

    extension = extout.lower()

    if extension == '.png':
        from matplotlib.backends.backend_agg \
            import FigureCanvasAgg as FigureCanvas
    elif extension == '.pdf':
        from matplotlib.backends.backend_pdf \
            import FigureCanvasPDF as FigureCanvas
    elif extension == '.svg':
        from matplotlib.backends.backend_svg \
            import FigureCanvasSVG as FigureCanvas
    else:
        raise ValueError("Can only handle extensions 'png', 'svg' or 'pdf'")

    from matplotlib.figure import Figure
    fig = Figure(figsize=(width, height), dpi=dpi)
    canvas = FigureCanvas(fig)

    ax = fig.add_axes([0, 0, 1, 1], aspect='auto',
                      frameon=False, xticks=[], yticks=[])

    basename, ext = os.path.splitext(basename)
    ax.imshow(im, aspect='auto', resample=True,
              interpolation='bilinear')
    fig.savefig(thumbfile, dpi=dpi)
    return fig


class ExecFile:
    """Execute the file and store the output, docstring, and
    sequence of matplotlib figures
    """
    def __init__(self, filename, execute=True, print_output=False):
        self.filename = filename
        self.extract_docstring()
        self.figlist = []
        self.output = ''
        self.stdout = sys.stdout
        self.print_output = print_output
        if execute:
            self.execute_file()

    def save_figures(self, fmt, thumb_fmt=None, scale=0.2,
                     max_width=200, max_height=150,
                     interpolation='bilinear', **savefig_kwds):
        from matplotlib import colors

        figlist = []

        for fig in self.figlist:
            figfile = fmt % fig.number
            print("saving", figfile)

            # if black background, save with black background as well.
            if colors.colorConverter.to_rgb(fig.get_facecolor()) == (0, 0, 0):
                fig.savefig(figfile,
                            facecolor='k',
                            edgecolor='none',
                            **savefig_kwds)
            else:
                fig.savefig(figfile, **savefig_kwds)
            plt.close(fig)

            if thumb_fmt is not None:
                thumbfile = thumb_fmt % fig.number
                fig = create_thumbnail(figfile, thumbfile,
                                       scale, max_width, max_height,
                                       interpolation)
                plt.close(fig)

            figlist.append(figfile)

        return figlist

    def write(self, s):
        if self.print_output:
            self.stdout.write(s)
        self.output += s

    def flush(self):
        if self.print_output:
            self.stdout.flush()

    def extract_docstring(self):
        """ Extract a module-level docstring
        """
        lines = open(self.filename).readlines()
        start_row = 0
        if lines[0].startswith('#!'):
            lines.pop(0)
            start_row = 1

        docstring = ''
        first_par = ''

        tokens = tokenize.generate_tokens(lines.__iter__().__next__)
        for tok_type, tok_content, _, (erow, _), _ in tokens:
            tok_type = token.tok_name[tok_type]
            if tok_type in ('NEWLINE', 'COMMENT', 'NL', 'INDENT', 'DEDENT'):
                continue
            elif tok_type == 'STRING':
                docstring = eval(tok_content)
                # If the docstring is formatted with several paragraphs, extract
                # the first one:
                paragraphs = '\n'.join(line.rstrip()
                                       for line in docstring.split('\n')).split('\n\n')
                if len(paragraphs) > 0:
                    first_par = paragraphs[0]
            break

        self.docstring = docstring
        self.short_desc = first_par
        self.end_line = erow + 1 + start_row

    def execute_file(self):
        """Execute the file, catching standard output
        and matplotlib figures
        """
        dirname, fname = os.path.split(self.filename)
        print('plotting %s' % fname)

        # close any currently open figures
        plt.close('all')

        # change to file directory for execution
        cwd = os.getcwd()

        try:
            if dirname:
                os.chdir(dirname)

            # set stdout to self in order to catch output (with write method)
            sys.stdout = self

            # execute the file
            with open(os.path.basename(self.filename)) as f:
                code = compile(f.read(), "somefile.py", 'exec')
                exec(code, {'pl': plt, 'plt': plt, 'pylab': plt})

            fig_mgr_list = matplotlib._pylab_helpers.Gcf.get_all_fig_managers()
            self.figlist = [manager.canvas.figure for manager in fig_mgr_list]

            self.figlist = sorted(self.figlist,
                                  key = lambda fig: fig.number)

        except:
            print(80 * '_')
            print('%s is not compiling:' % fname)
            traceback.print_exc()
            print(80 * '_')
        finally:
            # change back to original directory, and reset sys.stdout
            sys.stdout = self.stdout
            os.chdir(cwd)
            ncol = gc.collect()
            if self.print_output and (ncol > 0):
                print("\n > collected %i unreachable objects" % ncol)


class ExampleBuilder:
    """Example Builder for sphinx documentation"""
    def __init__(self, source_dir, target_dir,
                 image_dir='images', thumb_dir='images',
                 stdout_dir='.', script_dir='.', rst_dir='.',
                 sphinx_tag_base='example',
                 template_example=None,
                 template_index=None,
                 contents_file=None,
                 dir_info_file=None,
                 dir_footer_file=None,
                 execute_files=True,
                 force_rerun=False):
        self.source_dir = source_dir
        self.target_dir = target_dir

        self.image_dir = image_dir
        self.thumb_dir = thumb_dir
        self.stdout_dir = stdout_dir
        self.script_dir = script_dir
        self.rst_dir = rst_dir

        self.sphinx_tag_base = sphinx_tag_base

        self.contents_file = contents_file
        self.dir_info_file = dir_info_file
        self.dir_footer_file = dir_footer_file

        self.force_rerun = force_rerun
        self.execute_files = execute_files

        if template_example is None:
            self.template_example = DEFAULT_RST_TEMPLATE
        else:
            self.template_example = template_example

        if template_index is None:
            self.template_index = DEFAULT_INDEX_TEMPLATE
        else:
            self.template_index = template_index

    def read_contents(self, path, check_for_missing=True):
        """Read contents file

        A contents file is a list of filenames within the directory,
        with comments marked by '#' in the normal way.

        Parameters
        ----------
        path : str
            directory path *relative to self.target_dir* where the
            contents are located.  The file `self.contents_file`
            at this location will be read

        Returns
        -------
        L : list
            list of filenames from the contents
        """
        contents_dir = os.path.join(self.source_dir, path)
        contents_file = os.path.join(contents_dir, self.contents_file)
        return read_contents(contents_file, check_for_missing)

    # ============================================================
    # Directory parser:
    #
    #  This takes a path *relative to source_dir* and parses the
    #  contents, returning two lists: `scripts` & `subdirs`.  `scripts`
    #  should contain all executable scripts, and `subdirs` should contain
    #  all valid sub-directories.  This can be overloaded as needed.
    def parse_directory(self, path):
        dirpath = os.path.join(self.source_dir, path)
        if self.contents_file is None:
            L = os.listdir(dirpath)
        else:
            L = self.read_contents(path)

        subdirs = [d for d in L if os.path.isdir(os.path.join(dirpath, d))]
        scripts = [s for s in L if s.endswith('.py')]

        return subdirs, scripts

    # ======================================================================
    # Path Definitions
    #
    #  Each of these takes an example path *relative to source_dir* and
    #  returns a path *relative to target_dir*.  They can be overloaded
    #  as needed
    def image_filename(self, example_path):
        filename, ext = os.path.splitext(example_path)
        filename += '_%s'
        return os.path.join(self.image_dir, filename + '.png')

    def thumb_filename(self, example_path):
        filename, ext = os.path.splitext(example_path)
        filename += '_%s'
        return os.path.join(self.thumb_dir, filename + '_thumb.png')

    def stdout_filename(self, example_path):
        filename, ext = os.path.splitext(example_path)
        return os.path.join(self.stdout_dir, filename + '.txt')

    def rst_filename(self, example_path):
        filename, ext = os.path.splitext(example_path)
        return os.path.join(self.rst_dir, filename + '.rst')

    def html_filename(self, example_path):
        filename, ext = os.path.splitext(example_path)
        return filename + '.html'

    def rst_index_filename(self, dir_path):
        return os.path.join(self.rst_dir, dir_path, 'index.rst')

    def py_filename(self, example_path):
        return example_path

    def sphinx_tag(self, example_path):
        tag = os.path.splitext(os.path.normpath(example_path))[0]
        tag = tag.replace('/', '_')
        if tag in ('', '.'):
            tag = 'root'
        return '_'.join([self.sphinx_tag_base, tag])

    # ============================================================
    # RST generation scripts
    def run(self):
        self.generate_dir_rst('')

    def image_list(self, figlist):
        if len(figlist) == 0:
            imlist = ""

        elif len(figlist) == 1:
            imlist = ("\n"
                      ".. image:: %s\n"
                      "    :scale: 100\n"
                      "    :align: center\n" % figlist[0])

        else:
            imlist = "\n.. rst-class:: horizontal\n"
            for fig in figlist:
                imlist += ('\n\n'
                           '.. image:: %s\n'
                           '    :align: center\n'
                           '    :scale: 100\n\n' % fig)

        return imlist

    def figure_contents(self, path, filelist):
        toctree = ("\n\n"
                   ".. toctree::\n"
                   "   :hidden:\n\n")
        contents = "\n\n"

        for f in filelist:
            f = os.path.join(path, f)
            rel_thumb = os.path.relpath(self.thumb_filename(f) % 1, path)
            rel_html = os.path.relpath(self.html_filename(f), path)

            toctree += "   ./%s\n\n" % os.path.splitext(rel_html)[0]

            contents += (".. figure:: ./%s\n"
                         "    :target: ./%s\n"
                         "    :align: center\n"
                         "\n"
                         "    :ref:`%s`\n\n" % (rel_thumb,
                                                rel_html,
                                                self.sphinx_tag(f)))
        return toctree + contents

    def subdir_contents(self, path, subdirs):
        subdirs = [os.path.join(path, subdir) for subdir in subdirs]

        subdir_contents = ("\n\n"
                           ".. toctree::\n"
                           "   :maxdepth: 2\n\n")

        for subdir in subdirs:
            index = os.path.splitext(self.rst_index_filename(subdir))[0]
            subdir_contents += '   %s\n' % os.path.relpath(index, path)

        subdir_contents += '\n'
        return subdir_contents

    def generate_dir_rst(self, path):
        """Create an RST file for a directory

        This will also call generate_example_rst() for every python
        file in the directory

        Parameters
        ----------
        path : str
            path to the directory *relative to self.source_dir*
        """
        # Get information from the README or other specified file
        try:
            info = open(os.path.join(self.source_dir, path,
                                     self.dir_info_file)).read()
        except:
            info = os.path.split(path.rstrip('/'))[-1]
            info += '\n' + (len(info) * '-') + '\n'

        # Get the footer information
        try:
            footer = open(os.path.join(self.source_dir, path,
                                       self.dir_footer_file)).read()
        except:
            footer = ''

        # Find the index file
        index_file = os.path.join(self.target_dir,
                                  self.rst_index_filename(path))
        if not os.path.exists(os.path.split(index_file)[0]):
            os.makedirs(os.path.split(index_file)[0])

        # Get sub-directories and scripts
        subdirs, scripts = self.parse_directory(path)

        # Run the scripts
        for script in scripts:
            self.generate_example_rst(os.path.join(path, script))

        outfile = open(index_file, 'w')
        outfile.write(self.template_index %
                      dict(sphinx_tag=self.sphinx_tag(path),
                           subdir_contents=self.subdir_contents(path, subdirs),
                           info=info,
                           footer=footer,
                           figure_contents=self.figure_contents(path,
                                                                scripts)))

        # Recursively probe the sub-directories
        for subdir in subdirs:
            self.generate_dir_rst(os.path.join(path, subdir))

    def generate_example_rst(self, path):
        """Execute file, save images and standard out, and create RST file

        Parameters
        ----------
        path : str
            path to the example file *relative to self.source_dir*
        """
        example_file = os.path.join(self.source_dir, path)

        stdout_file = os.path.join(self.target_dir, self.stdout_filename(path))
        image_file = os.path.join(self.target_dir, self.image_filename(path))
        thumb_file = os.path.join(self.target_dir, self.thumb_filename(path))
        rst_file = os.path.join(self.target_dir, self.rst_filename(path))
        py_file = os.path.join(self.target_dir, self.py_filename(path))

        # check if we should re-run the example
        if os.path.exists(stdout_file):
            output_modtime = os.stat(stdout_file).st_mtime
            source_modtime = os.stat(example_file).st_mtime

            if output_modtime >= source_modtime:
                run_example = self.force_rerun and self.execute_files
            else:
                run_example = self.execute_files
        else:
            run_example = self.execute_files

        EF = ExecFile(example_file, execute=run_example)

        if run_example:
            # make sure directories exist
            for f in (stdout_file, image_file, thumb_file, rst_file, py_file):
                d, f = os.path.split(f)
                if not os.path.exists(d):
                    os.makedirs(d)

            # write output
            open(stdout_file, 'w').write(EF.output)

            # save all figures & thumbnails
            figure_list = EF.save_figures(image_file, thumb_file)

            # if no figures are created, we need to make a
            # blank thumb file
            if len(figure_list) == 0:
                shutil.copy('images/blank_image.png', thumb_file % 1)

        else:
            figure_list = list(glob.glob(image_file % '[1-9]'))

        # copy source code to generated file tree
        shutil.copyfile(example_file, py_file)

        figure_list = [os.path.relpath(f, os.path.join(self.target_dir,
                                                       os.path.dirname(path)))
                       for f in figure_list]

        fname = os.path.split(path)[1]
        stdout = os.path.split(stdout_file)[1]
        docstring = EF.docstring
        if docstring == '':
            docstring = '\n'.join([fname, '-' * len(fname), ''])

        rst_file = open(rst_file, 'w')
        rst_file.write(self.template_example %
                       dict(sphinx_tag=self.sphinx_tag(path),
                            docstring=docstring,
                            stdout=stdout,
                            fname=fname,
                            image_list=self.image_list(figure_list),
                            end_line=EF.end_line))


def read_contents(contents_file, check_for_missing=True):
    """Read contents file

    A contents file is a list of filenames within the directory,
    with comments marked by '#' in the normal way.

    Parameters
    ----------
    contents_file : str
        location of the contents file

    Returns
    -------
    L : list
        list of filenames from the contents
    """
    if not os.path.exists(contents_file):
        warnings.warn("Contents file not found in {}. "
                      "Skipping directory".format(contents_file))
        return []

    if check_for_missing:
        contents_dir = os.path.split(contents_file)[0]
        files = os.listdir(contents_dir)
    else:
        contents_dir = None
        files = None

    L = []
    for line in open(contents_file):
        filename = line.split('#')[0].strip()
        if len(filename) == 0:
            continue

        if check_for_missing and (filename not in files):
            raise ValueError("Fatal: file %s not found in %s"
                             % (filename, contents_dir))

        L.append(filename)

    return L
