import os
import fnmatch
from contextlib import contextmanager
from nose import SkipTest

import matplotlib
from matplotlib import colors
matplotlib.use('Agg')

# set some font properties
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif', style='normal', variant='normal',
              stretch='normal', weight='normal',)

import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
from matplotlib.testing.noseclasses import ImageComparisonFailure


KNOWN_FAILURES = [

    # System-dependent benchmark stuff
    'book_figures/chapter2/fig_sort_scaling.py',
    'book_figures/chapter2/fig_search_scaling.py',

    # MCMC (why doesn't random seed work???
    'book_figures/chapter5/fig_cauchy_mcmc.py',
    'book_figures/chapter5/fig_signal_background.py',
    'book_figures/chapter5/fig_model_comparison_mcmc.py',
    'book_figures/chapter5/fig_gaussgauss_mcmc.py',
    'book_figures/chapter8/fig_outlier_rejection.py',
    'book_figures/chapter10/fig_arrival_time.py',
    'book_figures/chapter10/fig_matchedfilt_burst.py',
    'book_figures/chapter10/fig_matchedfilt_chirp.py',
    'book_figures/chapter10/fig_matchedfilt_chirp2.py',
]


@contextmanager
def set_cwd(directory):
    cwd = os.getcwd()
    if directory: os.chdir(directory)
    yield
    os.chdir(cwd) 


def check_figure(filename, tol=1E-3):
    """
    Compare the given filename to the baseline
    """
    file_fmt = os.path.splitext(filename)[0] + "_{0}.png"

    dirname, fname = os.path.split(filename)

    baseline = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            'baseline', file_fmt))
    result = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          'results', file_fmt))

    resultdir = os.path.dirname(result)
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    plt.close('all')

    with set_cwd(dirname):
        with open(fname) as f:
            code = compile(f.read(), "somefile.py", 'exec')
            exec(code, {'pl' : plt, 'plt' : plt, 'pylab' : plt})

    for fignum in plt.get_fignums():
        fig = plt.figure(fignum)

        if colors.colorConverter.to_rgb(fig.get_facecolor()) == (0, 0, 0):
            fig.savefig(result.format(fignum), facecolor='k', edgecolor='none')
        else:
            fig.savefig(result.format(fignum))

        err = compare_images(baseline.format(fignum),
                             result.format(fignum),
                             tol)
        if err:
            if filename in KNOWN_FAILURES:
                raise SkipTest("known errors in {0}".format(filename))
            else:
                raise ImageComparisonFailure(err)


def test_book_figures(tol=0.1):
    cwd = os.getcwd()

    for chapter in os.listdir('book_figures'):
        if not os.path.isdir(os.path.join(cwd, 'book_figures', chapter)):
            continue
        for filename in os.listdir(os.path.join(cwd, 'book_figures', chapter)):
            if not fnmatch.fnmatch(filename, "fig_*.py"):
                continue
            os.chdir(cwd)
            filename = os.path.join('book_figures', chapter, filename)
            yield check_figure, filename, tol
