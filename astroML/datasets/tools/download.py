from __future__ import print_function, division

import sys

from ...py3k_compat import urlopen, BytesIO, url_content_length


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
        if true, return a BytesIO buffer rather than a string

    Returns
    -------
    s : string
        content of the file
    """
    num_units = 40

    fhandle = urlopen(data_url)
    content_length = url_content_length(fhandle)
    
    chunk_size = content_length // num_units

    print("Downloading %s" % data_url)
    nchunks = 0
    buf = BytesIO()
    content_length_str = bytes_to_string(content_length)

    while True:
        next_chunk = fhandle.read(chunk_size)
        nchunks += 1

        if next_chunk:
            buf.write(next_chunk)
            s = ('[' + nchunks * '='
                 + (num_units - 1 - nchunks) * ' '
                 + ']  %s / %s   \r' % (bytes_to_string(buf.tell()),
                                        content_length_str))
        else:
            sys.stdout.write('\n')
            break

        sys.stdout.write(s)
        sys.stdout.flush()

    buf.seek(0)
    if return_buffer:
        return buf
    else:
        return buf.getvalue()
