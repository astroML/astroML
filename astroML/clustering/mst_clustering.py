"""
Minimum Spanning Tree Clustering
"""
import numpy as np

from scipy import sparse
from sklearn.neighbors import kneighbors_graph
from sklearn.mixture import GMM

try:
    from scipy.sparse.csgraph import \
        minimum_spanning_tree, connected_components
except:
    raise ValueError("scipy v0.11 or greater required "
                     "for minimum spanning tree")


class HierarchicalClustering(object):
    """Hierarchical Clustering via Approximate Euclidean Minimum Spanning Tree

    Parameters
    ----------
    n_neighbors : int
        number of neighbors of each point used for approximate Euclidean
        minimum spanning tree (MST) algorithm.  See Notes below.
    edge_cutoff : float
        specify a fraction of edges to keep when selecting clusters.
        edge_cutoff should be between 0 and 1.
    min_cluster_size : int, optional
        specify a minimum number of points per cluster.  If not specified,
        all clusters will be kept.

    Attributes
    ----------
    X_train_ : ndarray
        the training data
    full_tree_ : sparse graph
        the full approximate Euclidean MST spanning the data
    cluster_graph_ : sparse graph
        the final (truncated) graph showing clusters
    n_components_ : int
        the number of clusters found.
    labels_ : int
        the cluster labels for each training point.  Labels range from -1
        to n_components_ - 1: points labeled -1 are in the background (i.e.
        their clusters were smaller than min_cluster_size)

    Notes
    -----
    This routine uses an approximate Euclidean minimum spanning tree (MST)
    to perform hierarchical clustering.  A true Euclidean minimum spanning
    tree naively costs O[N^3].  Graph traversal algorithms only help so much,
    because all N^2 edges must be used as candidates.  In this approximate
    algorithm, we use k < N edges from each point, so that the cost is only
    O[Nk log(Nk)]. For k = N, the approximation is exact; in practice for
    well-behaved data sets, the result is exact for k << N.
"""
    def __init__(self, n_neighbors=20,
                 edge_cutoff=0.9,
                 min_cluster_size=1):
        self.n_neighbors = n_neighbors
        self.edge_cutoff = edge_cutoff
        self.min_cluster_size = min_cluster_size

    def fit(self, X):
        """Fit the clustering model

        Parameters
        ----------
        X : array_like
            the data to be clustered: shape = [n_samples, n_features]
        """
        X = np.asarray(X, dtype=float)

        self.X_train_ = X

        # generate a sparse graph using the k nearest neighbors of each point
        G = kneighbors_graph(X, n_neighbors=self.n_neighbors, mode='distance')

        # Compute the minimum spanning tree of this graph
        self.full_tree_ = minimum_spanning_tree(G, overwrite=True)

        # Find the cluster labels
        self.n_components_, self.labels_, self.cluster_graph_ =\
            self.compute_clusters()

        return self

    def compute_clusters(self, edge_cutoff=None, min_cluster_size=None):
        """Compute the clusters given a trained tree

        After fit() is called, this method may be called to obtain a
        clustering result with a new edge_cutoff and min_cluster_size.

        Parameters
        ----------
        edge_cutoff : float, optional
            specify a fraction of edges to keep when selecting clusters.
            edge_cutoff should be between 0 and 1.  If not specified,
            self.edge_cutoff will be used.
        min_cluster_size : int, optional
            specify a minimum number of points per cluster.  If not specified,
            self.min_cluster_size will be used.

        Returns
        -------
        n_components : int
            the number of clusters found
        labels : ndarray
            the labels of each point.  Labels range from -1 to
            n_components_ - 1: points labeled -1 are in the background
            (i.e. their clusters were smaller than min_cluster_size)
        T_trunc : sparse matrix
            the truncated minimum spanning tree
        """
        if edge_cutoff is None:
            edge_cutoff = self.edge_cutoff

        if min_cluster_size is None:
            min_cluster_size = self.min_cluster_size

        if not hasattr(self, 'full_tree_'):
            raise ValueError("must call fit() before calling "
                             "compute_clusters()")

        T_trunc = self.full_tree_.copy()

        # cut-off edges at the percentile given by edge_cutoff
        cutoff = np.percentile(T_trunc.data, 100 * edge_cutoff)
        T_trunc.data[T_trunc.data > cutoff] = 0
        T_trunc.eliminate_zeros()

        # find connected components
        n_components, labels = connected_components(T_trunc, directed=False)
        counts = np.bincount(labels)

        # for all components with less than min_cluster_size points, set
        # to background, and re-label the clusters
        i_bg = np.where(counts < min_cluster_size)[0]

        for i in i_bg:
            labels[labels == i] = -1

        if len(i_bg) > 0:
            _, labels = np.unique(labels, return_inverse=True)
            labels -= 1
            n_components = labels.max() + 1

        # eliminate links in T_trunc which are not clusters
        I = sparse.eye(len(labels), len(labels))
        I.data[0, labels < 0] = 0
        T_trunc = I * T_trunc * I

        return n_components, labels, T_trunc


def get_graph_segments(X, G):
    """Get graph segments for plotting a 2D graph

    Parameters
    ----------
    X : array_like
        the data, of shape [n_samples, 2]
    G : array_like or sparse graph
        the [n_samples, n_samples] matrix encoding the graph of connectinons
        on X

    Returns
    -------
    x_coords, y_coords : ndarrays
        the x and y coordinates for plotting the graph.  They are of size
        [2, n_links], and can be visualized using
        ``plt.plot(x_coords, y_coords, '-k')``
    """
    X = np.asarray(X)
    if (X.ndim != 2) or (X.shape[1] != 2):
        raise ValueError('shape of X should be (n_samples, 2)')

    n_samples = X.shape[0]

    G = sparse.coo_matrix(G)
    A = X[G.row].T
    B = X[G.col].T

    x_coords = np.vstack([A[0], B[0]])
    y_coords = np.vstack([A[1], B[1]])

    return x_coords, y_coords
