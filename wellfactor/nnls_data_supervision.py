import math
import numpy as np
import scipy.sparse as sps
import numpy.linalg as nla
import scipy.linalg as sla

from wellfactor.nnls import nnlsm_blockpivot

def nnlsm_blockpivot_data_supervised(W, A, for_H=True, is_input_prod=False, init=None, 
        feature_split_points = [], data_split_points = [], normA=0):
    if for_H == True:
        H1 = init[:,:data_split_points[0]]
        H2 = init[:,data_split_points[0]:data_split_points[1]]
        H3 = init[:,data_split_points[1]:data_split_points[2]]
        H4 = init[:,data_split_points[2]:data_split_points[3]]
        A1 = A[:,:data_split_points[0]]
        A2 = A[:feature_split_points[1], data_split_points[0]:data_split_points[1]]
        A3 = A[list(range(feature_split_points[0])) + \
               list(range(feature_split_points[1],feature_split_points[2])),\
                data_split_points[1]:data_split_points[2]]
        A4 = A[:feature_split_points[0],data_split_points[2]:data_split_points[3]]

        WtW1 = W.T.dot(W)
        W2 = W[:feature_split_points[1],:]
        WtW2 = W2.T.dot(W2)
        W3 = W[list(range(feature_split_points[0])) + \
               list(range(feature_split_points[1],feature_split_points[2])),:]
        WtW3 = W3.T.dot(W3)
        W4 = W[:feature_split_points[0],:]
        WtW4 = W4.T.dot(W4)
        if sps.issparse(A):
            WtA1 = (A1.T.dot(W)).T
            WtA2 = (A2.T.dot(W2)).T
            WtA3 = (A3.T.dot(W3)).T
            WtA4 = (A4.T.dot(W4)).T
        else:
            WtA1 = W.T.dot(A1)
            WtA2 = W2.T.dot(A2)
            WtA3 = W3.T.dot(A3)
            WtA4 = W4.T.dot(A4)
        Sol1, _ = nnlsm_blockpivot(WtW1, WtA1, init=H1, is_input_prod=True)
        Sol2, _ = nnlsm_blockpivot(WtW2, WtA2, init=H2, is_input_prod=True)
        Sol3, _ = nnlsm_blockpivot(WtW3, WtA3, init=H3, is_input_prod=True)
        Sol4, _ = nnlsm_blockpivot(WtW4, WtA4, init=H4, is_input_prod=True)
        return np.concatenate((Sol1,Sol2,Sol3,Sol4),axis=1), 0
    else:
        H1 = init[:,:feature_split_points[0]]
        H2 = init[:,feature_split_points[0]:feature_split_points[1]]
        H3 = init[:,feature_split_points[1]:feature_split_points[2]]
        A1 = A[:,:feature_split_points[0]]
        A2 = A[:data_split_points[1], feature_split_points[0]:feature_split_points[1]]
        A3 = A[list(range(data_split_points[0])) + \
               list(range(data_split_points[1],data_split_points[2])),\
                feature_split_points[1]:feature_split_points[2]]

        WtW1 = W.T.dot(W)
        W2 = W[:data_split_points[1],:]
        WtW2 = W2.T.dot(W2)
        W3 = W[list(range(data_split_points[0])) + \
                list(range(data_split_points[1],data_split_points[2])),:]
        WtW3 = W3.T.dot(W3)
        if sps.issparse(A):
            WtA1 = (A1.T.dot(W)).T
            WtA2 = (A2.T.dot(W2)).T
            WtA3 = (A3.T.dot(W3)).T
        else:
            WtA1 = W.T.dot(A1)
            WtA2 = W2.T.dot(A2)
            WtA3 = W3.T.dot(A3)
        Sol1, _ = nnlsm_blockpivot(WtW1, WtA1, init=H1, is_input_prod=True)
        Sol2, _ = nnlsm_blockpivot(WtW2, WtA2, init=H2, is_input_prod=True)
        Sol3, _ = nnlsm_blockpivot(WtW3, WtA3, init=H3, is_input_prod=True)

        sum_squared = normA * normA \
            - 2 * np.trace(WtA1.dot(Sol1.T)) \
            - 2 * np.trace(WtA2.dot(Sol2.T)) \
            - 2 * np.trace(WtA3.dot(Sol3.T)) \
            + np.trace((WtW1).dot(Sol1.dot(Sol1.T))) \
            + np.trace((WtW2).dot(Sol2.dot(Sol2.T))) \
            + np.trace((WtW3).dot(Sol3.dot(Sol3.T)))
        return np.concatenate((Sol1,Sol2,Sol3),axis=1), math.sqrt(np.maximum(sum_squared,0))

if __name__ == '__main__':
    pass
