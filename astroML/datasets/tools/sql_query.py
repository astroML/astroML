"""
Tools to perform a SQL queries to an online server.
Default values are provided for http://cas.sdss.org
"""
from ...py3k_compat import urlencode, urlopen

PUBLIC_URL = 'http://cas.sdss.org/public/en/tools/search/x_sql.asp'
DEFAULT_FMT = 'csv'


def remove_sql_comments(sql):
    """Strip SQL comments starting with --"""
    return ' \n'.join(map(lambda x: x.split('--')[0], sql.split('\n')))


def sql_query(sql_str, url=PUBLIC_URL, format='csv'):
    """Execute query

    Parameters
    ----------
    sql_str : string
        valid sql query

    url: string (optional)
        query url.  Default is http://cas.sdss.org query script

    format: string (default='csv')
        query output format

    Returns
    -------
    F: file object
        results of the query
    """
    sql_str = remove_sql_comments(sql_str)
    params = urlencode(dict(cmd=sql_str, format=format))
    return urlopen(url + '?%s' % params)
