import numpy as np
import scipy.sparse as sps
from wellfactor.matrix_utils import norm_fro, norm_fro_err, column_norm, normalize_column_pair, normalize_column, sparse_remove_row, sparse_remove_column, NormType

def test_norm_fro_err():
    X = np.array([[2.0, 5.0, 0.0], [-3.0, 0.0, 1.5]])
    W = np.array([[1.0], [2.0]])
    H = np.array([[1.0], [1.0], [1.0]])
    norm_X_fro = norm_fro(X)

    val1 = norm_fro(X - W.dot(H.T))
    val2 = norm_fro_err(X, W, H, norm_X_fro)
    
    assert np.isclose(val1, val2)

def test_column_norm():
    X = np.array([[2.0, 5.0, 0.0], [-3.0, 0.0, 1.5]])
    val1 = column_norm(X, by_norm=NormType.L2)
    val2 = np.sqrt(np.array([4 + 9, 25, 1.5 * 1.5]))
    assert np.allclose(val1, val2)

def test_normalize_column_pair():
    W = np.array([[1.0, -2.0], [2.0, 3.0]])
    H = np.array([[-0.5, 1.0], [1.0, 2.0], [1.0, 0.0]])
    val1 = column_norm(W, by_norm=NormType.L2)
    val3 = W.dot(H.T)
    W1, H1, weights = normalize_column_pair(W, H, by_norm=NormType.L2)
    val2 = column_norm(W1, by_norm=NormType.L2)
    val4 = W1.dot(H1.T)
    assert np.allclose(val1, weights)
    assert np.allclose(val2, np.array([1.0, 1.0]))
    assert np.allclose(val3, val4)

def test_normalize_column():
    X = np.array([[2.0, 5.0, 0.0], [-3.0, 0.0, 1.5]])
    val1 = column_norm(X, by_norm=NormType.L2)
    X1, weights = normalize_column(X, by_norm=NormType.L2)
    val2 = column_norm(X1, by_norm=NormType.L2)
    assert np.allclose(val2, np.array([1.0, 1.0, 1.0]))
    assert np.allclose(val1, weights)
    assert np.allclose(X.shape, X1.shape)

def test_sparse_remove_row():
    X = sps.csr_matrix(np.array([[2.0, 5.0, 0.0], [-3.0, 0.0, 1.5], [0.5, -2.0, 2.5]]))
    X1 = sparse_remove_row(X, [1]).todense()
    val1 = np.array([[2.0, 5.0, 0.0], [0.5, -2.0, 2.5]])
    assert np.allclose(X1, val1)

def test_sparse_remove_column():
    X = sps.csr_matrix(np.array([[2.0, 5.0, 0.0], [-3.0, 0.0, 1.5], [0.5, -2.0, 2.5]]))
    X1 = sparse_remove_column(X, [1]).todense()
    val1 = np.array([[2.0, 0.0], [-3.0, 1.5], [0.5, 2.5]])
    assert np.allclose(X1, val1)
