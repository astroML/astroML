def setup_text_plots(fontsize=8, usetex=True):
    """
    This function adjusts matplotlib settings so that all figures in the
    textbook have a uniform format and look.
    """
    import matplotlib
    from distutils.version import LooseVersion
    matplotlib.rc('legend', fontsize=fontsize, handlelength=3)
    matplotlib.rc('axes', titlesize=fontsize)
    matplotlib.rc('axes', labelsize=fontsize)
    matplotlib.rc('xtick', labelsize=fontsize)
    matplotlib.rc('ytick', labelsize=fontsize)
    matplotlib.rc('text', usetex=usetex)
    matplotlib.rc('font', size=fontsize, family='serif',
                  style='normal', variant='normal',
                  stretch='normal', weight='normal')
    matplotlib.rc('patch', force_edgecolor=True)
    if LooseVersion(matplotlib.__version__) < LooseVersion("3.1"):
        matplotlib.rc('_internal', classic_mode=True)
    else:
        # New in mpl 3.1
        matplotlib.rc('scatter', edgecolors='b')
    matplotlib.rc('grid', linestyle=':')
    matplotlib.rc('errorbar', capsize=3)
    matplotlib.rc('image', cmap='viridis')
    matplotlib.rc('axes', xmargin=0)
    matplotlib.rc('axes', ymargin=0)
    matplotlib.rc('xtick', direction='in')
    matplotlib.rc('ytick', direction='in')
    matplotlib.rc('xtick', top=True)
    matplotlib.rc('ytick', right=True)
