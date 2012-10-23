import numpy as np
from numpy.testing import assert_, assert_allclose
from astroML.clustering import HierarchicalClustering, get_graph_segments


def test_simple_clustering():
    np.random.seed(0)
    N = 10
    X = np.random.random((N, 2))
    model = HierarchicalClustering(8, edge_cutoff=0.5)
    model.fit(X)

    assert_(model.n_components_ == N / 2)
    assert_(np.sum(model.full_tree_.toarray() > 0) == N - 1)
    assert_(np.sum(model.cluster_graph_.toarray() > 0) == N / 2)
    assert_allclose(np.unique(model.labels_), np.arange(N / 2))


def test_cluster_cutoff():
    np.random.seed(0)
    N = 100
    X = np.random.random((N, 2))
    model = HierarchicalClustering(8, edge_cutoff=0.9, min_cluster_size=10)
    model.fit(X)

    assert_allclose(np.unique(model.labels_),
                    np.arange(-1, model.n_components_))


def test_graph_segments():
    np.random.seed(0)
    N = 4
    X = np.random.random((N, 2))
    G = np.zeros([N, N])
    G[0, 1] = 1
    G[2, 1] = 1
    G[2, 3] = 1

    ind = np.array([[0, 2, 2],
                    [1, 1, 3]])
    xseg_check = X[ind, 0]
    yseg_check = X[ind, 1]

    xseg, yseg = get_graph_segments(X, G)

    assert_allclose(xseg, xseg_check)
    assert_allclose(yseg, yseg_check)
