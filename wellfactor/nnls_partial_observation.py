import numpy as np
import scipy.sparse as sps
import numpy.linalg as nla
import scipy.linalg as sla

from wellfactor.nnls import nnlsm_blockpivot

def nnlsm_blockpivot_partial_observation(W, A, init=None, observed_feat_num=None, observed_item_num=None):
    if observed_feat_num is None or observed_item_num is None:
        raise ValueError("observed_feat_num and observed_item_num must be provided.")

    # Split initial matrix and A into observed and unobserved parts
    H_observed = init[:,:observed_item_num]
    H_unobserved = init[:,observed_item_num:]
    A_observed = A[:,:observed_item_num]
    A_unobserved = A[:observed_feat_num,observed_item_num:]

    # Compute necessary products
    WtW = W.T.dot(W)
    WtW_unobserved = W[:observed_feat_num,:].T.dot(W[:observed_feat_num,:])
    
    if sps.issparse(A):
        WtA_observed = A_observed.T.dot(W)
        WtA_observed = WtA_observed.T
        WtA_unobserved = A_unobserved.T.dot(W[:observed_feat_num,:])
        WtA_unobserved = WtA_unobserved.T
    else:
        WtA_observed = W.T.dot(A_observed)
        WtA_unobserved = W[:observed_feat_num,:].T.dot(A_unobserved)

    # Perform NNLS block pivot for observed and unobserved parts separately
    Sol_observed, _ = nnlsm_blockpivot(WtW, WtA_observed, init=H_observed, is_input_prod=True)
    Sol_unobserved, _ = nnlsm_blockpivot(WtW_unobserved, WtA_unobserved, init=H_unobserved, is_input_prod=True)
    
    return np.concatenate((Sol_observed,Sol_unobserved),axis=1)

if __name__ == '__main__':
    pass
