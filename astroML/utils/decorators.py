import warnings
import functools
from distutils.version import LooseVersion

import numpy as np
import astropy
import pickle

from astroML.utils.exceptions import AstroMLDeprecationWarning

# We use functionality of the deprecated decorator from astropy that was
# added in v2.0.10 LTS and v3.1
av = astropy.__version__
ASTROPY_LT_31 = (LooseVersion(av) < LooseVersion("2.0.10") or
                 (LooseVersion("3.0") <= LooseVersion(av) and LooseVersion(av) < LooseVersion("3.1")))

__all__ = ['pickle_results', 'deprecated']


def pickle_results(filename=None, verbose=True):
    """Generator for decorator which allows pickling the results of a funcion

    Pickle is python's built-in object serialization.  This decorator, when
    used on a function, saves the results of the computation in the function
    to a pickle file.  If the function is called a second time with the
    same inputs, then the computation will not be repeated and the previous
    results will be used.

    This functionality is useful for computations which take a long time,
    but will need to be repeated (such as the first step of a data analysis).

    Parameters
    ----------
    filename : string (optional)
        pickle file to which results will be saved.
        If not specified, then the file is '<funcname>_output.pkl'
        where '<funcname>' is replaced by the name of the decorated function.
    verbose : boolean (optional)
        if True, then print a message to standard out specifying when the
        pickle file is written or read.

    Examples
    --------
    >>> @pickle_results('tmp.pkl', verbose=True)
    ... def f(x):
    ...     return x * x
    >>> f(4)
    @pickle_results: computing results and saving to 'tmp.pkl'
    16
    >>> f(4)
    @pickle_results: using precomputed results from 'tmp.pkl'
    16
    """
    def pickle_func(f, filename=filename, verbose=verbose):
        if filename is None:
            filename = '%s_output.pkl' % f.__name__

        def new_f(*args, **kwargs):
            try:
                D = pickle.load(open(filename, 'rb'))
                cache_exists = True
            except:
                D = {}
                cache_exists = False

            # simple comparison doesn't work in the case of numpy arrays
            Dargs = D.get('args')
            Dkwargs = D.get('kwargs')

            try:
                args_match = (args == Dargs)
            except:
                args_match = np.all([np.all(a1 == a2)
                                     for (a1, a2) in zip(Dargs, args)])

            try:
                kwargs_match = (kwargs == Dkwargs)
            except:
                kwargs_match = ((sorted(Dkwargs.keys())
                                 == sorted(kwargs.keys()))
                                and (np.all([np.all(Dkwargs[key]
                                                    == kwargs[key])
                                             for key in kwargs])))

            if (type(D) == dict and D.get('funcname') == f.__name__
                    and args_match and kwargs_match):
                if verbose:
                    print("@pickle_results: using precomputed "
                          "results from '%s'" % filename)
                retval = D['retval']

            else:
                if verbose:
                    print("@pickle_results: computing results "
                          "and saving to '%s'" % filename)
                    if cache_exists:
                        print("  warning: cache file '%s' exists" % filename)
                        print("    - args match:   %s" % args_match)
                        print("    - kwargs match: %s" % kwargs_match)
                retval = f(*args, **kwargs)

                funcdict = dict(funcname=f.__name__, retval=retval,
                                args=args, kwargs=kwargs)
                with open(filename, 'wb') as outfile:
                    pickle.dump(funcdict, outfile)

            return retval
        return new_f
    return pickle_func


if not ASTROPY_LT_31:
    from astropy.utils.decorators import deprecated
else:
    def deprecated(since, message='', alternative=None, **kwargs):

        def deprecate_function(func, message=message, since=since,
                               alternative=alternative):
            if message == '':
                message = ('Function {} has been deprecated since {}.'
                           .format(func.__name__, since))
                if alternative is not None:
                    message += '\n Use {} instead.'.format(alternative)

            @functools.wraps(func)
            def deprecated_func(*args, **kwargs):
                warnings.warn(message, AstroMLDeprecationWarning)
                return func(*args, **kwargs)
            return deprecated_func

        def deprecate_class(cls, message=message, since=since,
                            alternative=alternative):
            if message == '':
                message = ('Class {} has been deprecated since {}.'
                           .format(cls.__name__, since))
                if alternative is not None:
                    message += '\n Use {} instead.'.format(alternative)

            cls.__init__ = deprecate_function(cls.__init__, message=message)

            return cls

        def deprecate(obj):
            if isinstance(obj, type):
                return deprecate_class(obj)
            else:
                return deprecate_function(obj)

        return deprecate
