"""
Compatibility utilities for Python 2 & 3
"""

import sys
py3k = (sys.version_info[0] == 3)

#----------------------------------------------------------------------
# urllib stuff

if py3k:
    from urllib.request import urlopen
    from urllib.error import HTTPError
    from urllib.parse import urlencode
else:
    from urllib2 import urlopen
    from urllib2 import HTTPError
    from urllib import urlencode

def url_content_length(fhandle):
    if py3k:
        length = dict(fhandle.info())['Content-Length']
    else:
        length = fhandle.info().getheader('Content-Length')
    return int(length.strip())


#----------------------------------------------------------------------
# pickle stuff
if py3k:
    from pickle import load, dump
else:
    from cPickle import load, dump

#----------------------------------------------------------------------
# StringIO
if py3k:
    from io import StringIO, BytesIO
else:
    from cStringIO import StringIO
    from cStringIO import StringIO as BytesIO
