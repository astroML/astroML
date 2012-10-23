import sys
import urllib2
from cStringIO import StringIO


def bytes_to_string(nbytes):
    if nbytes < 1024:
        return '%ib' % nbytes

    nbytes /= 1024.
    if nbytes < 1024:
        return '%.1fkb' % nbytes

    nbytes /= 1024.
    if nbytes < 1024:
        return '%.2fMb' % nbytes

    nbytes /= 1024.
    return '%.1fGb' % nbytes


def download_with_progress_bar(data_url, return_buffer=False):
    """Download a file, showing progress

    Parameters
    ----------
    data_url : string
        web address
    return_buffer : boolean (optional)
        if true, return a StringIO buffer rather than a string

    Returns
    -------
    s : string
        content of the file
    """
    num_units = 40

    fhandle = urllib2.urlopen(data_url)
    total_size = int(fhandle.info().getheader('Content-Length').strip())
    chunk_size = total_size / num_units

    print "Downloading %s" % data_url
    nchunks = 0
    buf = StringIO()
    total_size_str = bytes_to_string(total_size)

    while True:
        next_chunk = fhandle.read(chunk_size)
        nchunks += 1

        if next_chunk:
            buf.write(next_chunk)
            s = ('[' + nchunks * '='
                 + (num_units - 1 - nchunks) * ' '
                 + ']  %s / %s   \r' % (bytes_to_string(buf.tell()),
                                        total_size_str))
        else:
            sys.stdout.write('\n')
            break

        sys.stdout.write(s)
        sys.stdout.flush()

    buf.reset()
    if return_buffer:
        return buf
    else:
        return buf.getvalue()
