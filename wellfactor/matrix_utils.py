import math
from enum import Enum
from typing import List, Tuple, Union

import numpy as np
import numpy.linalg as nla
import scipy.sparse as sps

class NormType(Enum):
    L1 = '1'
    L2 = '2'

def norm_fro(X: Union[np.ndarray, sps.csr_matrix]) -> float:
    """ 
    Compute the Frobenius norm of a matrix.
    """
    if sps.issparse(X):     # scipy.sparse array
        return math.sqrt(X.multiply(X).sum())
    else:                   # numpy array
        return nla.norm(X)

def norm_fro_err(X: Union[np.ndarray, sps.csr_matrix], W: np.ndarray, H: np.ndarray, norm_X: float) -> float:
    """ 
    Compute the approximation error in Frobenius norm.
    norm(X - W.dot(H.T)) is efficiently computed based on trace() expansion 
    when W and H are thin.
    """
    sum_squared = norm_X * norm_X - 2 * np.trace(H.T.dot(X.T.dot(W))) \
        + np.trace((W.T.dot(W)).dot(H.T.dot(H)))
    return math.sqrt(np.maximum(sum_squared, 0))

def norm_fro_err_mask(X: np.ndarray, W: np.ndarray, H: np.ndarray, norm_X: float, observed_feat_num: int, observed_item_num: int) -> float:
    sum_squared = norm_X * norm_X - 2 * np.trace(H.T.dot(X.T.dot(W))) \
        + np.trace((W.T.dot(W)).dot(H.T.dot(H))) \
        - np.trace((W[observed_feat_num:,:].T.dot(W[observed_feat_num:,:])).dot(H[observed_item_num:,:].T.dot(H[observed_item_num:,:])))
    return math.sqrt(np.maximum(sum_squared, 0))

def column_norm(X: Union[np.ndarray, sps.csr_matrix], by_norm: NormType = NormType.L2) -> np.ndarray:
    """ 
    Compute the norms of each column of a given matrix.
    """
    if sps.issparse(X):
        if by_norm == NormType.L2:
            norm_vec = np.sqrt(X.multiply(X).sum(axis=0))
        elif by_norm == NormType.L1:
            norm_vec = X.sum(axis=0)
        return np.asarray(norm_vec)[0]
    else:
        if by_norm == NormType.L2:
            norm_vec = np.sqrt(np.sum(X * X, axis=0))
        elif by_norm == NormType.L1:
            norm_vec = np.sum(X, axis=0)
        return norm_vec

def normalize_column_pair(W: np.ndarray, H: np.ndarray, by_norm: NormType = NormType.L2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ 
    Column normalization for a matrix pair.
    Scale the columns of W and H so that the columns of W have unit norms and 
    the product W.dot(H.T) remains the same.  The normalizing coefficients are 
    also returned.
    """
    norms = column_norm(W, by_norm=by_norm)

    to_normalize = norms > 0
    W[:, to_normalize] /= norms[to_normalize]
    H[:, to_normalize] *= norms[to_normalize]
    
    weights = np.ones(norms.shape)
    weights[to_normalize] = norms[to_normalize]
    return (W, H, weights)

def normalize_column(X: Union[np.ndarray, sps.csr_matrix], by_norm: NormType = NormType.L2) -> Tuple[Union[np.ndarray, sps.csr_matrix], np.ndarray]:
    """ 
    Column normalization.
    Scale the columns of X so that they have unit l2-norms.
    The normalizing coefficients are also returned.
    """
    if sps.issparse(X):
        weights = column_norm(X, by_norm)
        # construct a diagonal matrix
        dia = sps.diags([1.0 / w if w > 0 else 1.0 for w in weights])
        Y = X.dot(dia)
        return (Y, weights)
    else:
        norms = column_norm(X, by_norm)
        to_normalize = norms > 0
        X[:, to_normalize] /= norms[to_normalize]
        weights = np.ones(norms.shape)
        weights[to_normalize] = norms[to_normalize]
        return (X, weights)

def sparse_remove_row(X: sps.csr_matrix, to_remove: List[int]) -> sps.csr_matrix:
    """ 
    Delete rows from a sparse matrix.
    """
    if not sps.isspmatrix_lil(X):
        X = X.tolil()

    to_keep = [i for i in range(0, X.shape[0]) if i not in to_remove]
    Y = sps.vstack([X.getrowview(i) for i in to_keep])
    return Y

def sparse_remove_column(X: sps.csr_matrix, to_remove: List[int]) -> sps.csr_matrix:
    """ 
    Delete columns from a sparse matrix.
    """
    B = sparse_remove_row(X.transpose().tolil(), to_remove).tocoo().transpose()
    return B