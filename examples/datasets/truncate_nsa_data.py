"""
NASA Sloan Atlas dataset size reduction
---------------------------------------

The NASA Sloan Atlas dataset is contained in a ~0.5GB available at
http://www.nsatlas.org/data

This is a command-line script which takes the location of 'nsa_v0_1_2.fits'
as input, and outputs 'nsa_v0_1_2_reduced.npy', saving a particularly useful
subset of the attributes and reducing the size of the file by ~90%

The reduced file can be loaded using astroML.datasets.fetch_nasa_atlas();
this example is just for reference.

Usage:
% python truncate_nsa_data.py /path/to/nsa_v0_1_2.fits [outputfile]

if outputfile is not specified, the output will be saved in the same directory
as the input.
"""
import numpy as np
import pyfits

FIELDS = ['RA', 'DEC', 'PLATE', 'FIBERID', 'MJD', 'S2NSAMP', 'Z', 'ZDIST',
          'ZDIST_ERR', 'NMGY', 'NMGY_IVAR', 'RNMGY', 'ABSMAG', 'AMIVAR',
          'EXTINCTION', 'MTOL', 'B300', 'B1000', 'METS', 'MASS',
          'SERSIC_N', 'SERSIC_BA', 'PETROTH50', 'PETROTH90', 'VDISP',
          'D4000', 'D4000ERR']

for line in ['S2', 'HA', 'N2', 'HB', 'O1', 'O2', 'O3']:
    FIELDS += [line + 'FLUX', line + 'FLUXERR']


def truncate_nsa_data(nsa_datafile, outfile=None, fields=FIELDS):
    """Truncate the NSA datafile
    
    Parameters
    ----------
    nsa_datafile: string
        the location of the file 'nsa_v0_1_2.fits', which can be
        downloaded from http://www.nsatlas.org/data
    outfile: string (optional)
        the .npy file to which the output will be saved
        If not specified, it will be in the same directory as the input
    keep_fields: list of strings (optional)
        The fields in the file to keep.  Defaults are those used to create
        the file downloaded by astroML.datasets.fetch_nsa_data()
    """
    if outfile is None:
        outfile = nsa_datafile.rstrip('.fits') + '_reduced.npy'

    MB = 2. ** 20.
    
    # Load input fits file
    print "loading %s" % nsa_datafile
    hdulist = pyfits.open(nsa_datafile)
    data = hdulist[1].data
    print " - input size: %.1f MB" % (data.size * data.itemsize / MB)

    # Build output array
    new_dtype = [(key, data.dtype.fields[key][0]) for key in fields]
    xnew = np.empty(data.shape, dtype=new_dtype)
    for key in fields:
        xnew[key] = data[key]

    # Save output array to file
    print "saving to %s" % outfile
    print " - output size: %.1f MB" % (xnew.size * xnew.itemsize / MB)
    np.save(outfile, xnew)


if __name__ == '__main__':
    import sys
    try:
        inputfile = sys.argv[1]
    except:
        print __doc__
        sys.exit()

    try:
        outputfile = sys.argv[2]
    except:
        outputfile = None

    truncate_nsa_data(inputfile, outputfile)
