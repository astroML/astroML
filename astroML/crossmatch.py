import numpy as np
from scipy.spatial import cKDTree


def crossmatch(X1, X2, max_distance=np.inf):
    """Cross-match the values between X1 and X2

    By default, this uses a KD Tree for speed.

    Parameters
    ----------
    X1 : array_like
        first dataset, shape(N1, D)
    X2 : array_like
        second dataset, shape(N2, D)
    max_distance : float (optional)
        maximum radius of search.  If no point is within the given radius,
        then inf will be returned.

    Returns
    -------
    dist, ind: ndarrays
        The distance and index of the closest point in X2 to each point in X1
        Both arrays are length N1.
        Locations with no match are indicated by
        dist[i] = inf, ind[i] = N2
    """
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)

    N1, D = X1.shape
    N2, D2 = X2.shape

    if D != D2:
        raise ValueError('Arrays must have the same second dimension')

    kdt = cKDTree(X2)

    dist, ind = kdt.query(X1, k=1, distance_upper_bound=max_distance)

    return dist, ind
