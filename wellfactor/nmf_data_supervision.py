from typing import Optional, Tuple, Union
import json
import time

import numpy as np
import scipy.sparse as sps
from wellfactor.nnls_data_supervision import nnlsm_blockpivot_data_supervised
from wellfactor import matrix_utils as mu

class DataSupervisionNMF:
    """ Base class for NMF algorithms

    Specific algorithms need to be implemented by deriving from this class.
    """
    def __init__(self, default_max_iter=50, default_max_time=np.inf):
        self.set_default(default_max_iter, default_max_time)

    def set_default(self, default_max_iter: int, default_max_time: Optional[int]) -> None:
        self.default_max_iter = default_max_iter
        self.default_max_time = default_max_time

    def run(self, A: Union[np.ndarray, sps.spmatrix], k: int, 
            init: Optional[Tuple[np.ndarray, np.ndarray]] = None, 
            max_iter: Optional[int] = None, max_time: Optional[int] = None, verbose: int = 0, 
            fully_observed_feature_num=None, 
            observed_idx=None,
            feature_split_points = [], data_split_points = []) -> Tuple[np.ndarray, np.ndarray, dict]:
        """ Run a NMF algorithm

        Parameters
        ----------
        A : numpy.ndarray or scipy.sparse matrix, shape (m,n)
            Input matrix to factorize.
        k : int
            Target lower rank.

        Optional Parameters
        -------------------
        init : tuple of numpy.ndarray, optional
            (W_init, H_init) where W_init is numpy.ndarray of shape (m,k) and
            H_init is numpy.ndarray of shape (n,k). If provided, these values are used as initial values for NMF iterations.

        max_iter : int, optional
            Maximum number of iterations. If not provided, default maximum for each algorithm is used.
        max_time : int, optional
            Maximum amount of time in seconds. If not provided, default maximum for each algorithm is used.
        verbose : int, optional
            0 (default) - No debugging information is collected, but input and output information is printed on screen.
            -1 - No debugging information is collected, and nothing is printed on screen.
            1 (debugging/experimental purpose) - History of computation is returned. See 'rec' variable.
            2 (debugging/experimental purpose) - History of computation is additionally printed on screen.

        Returns
        -------
        Tuple of numpy.ndarray and numpy.ndarray and dict
            (W, H, rec)
            W : numpy.ndarray, shape (m,k)
                Obtained factor matrix.
            H : numpy.ndarray, shape (n,k)
                Obtained coefficient matrix.
            rec : dict
                Auxiliary information about the execution.

        Raises
        ------
        ValueError
            If input matrix A has wrong shape or data type.
        """
        if not isinstance(A, (np.ndarray, sps.spmatrix)):
            raise ValueError('Input matrix A should be a numpy.ndarray or scipy.sparse matrix.')
        if A.ndim != 2:
            raise ValueError('Input matrix A should be a 2-dimensional matrix.')
        if not isinstance(k, int):
            raise ValueError('Target lower rank k should be an integer.')
        if k <= 0:
            raise ValueError('Target lower rank k should be a positive integer.')
        if init is not None and not isinstance(init, tuple):
            raise ValueError('Initial values should be provided as a tuple (W_init, H_init).')
        if max_iter is not None and (not isinstance(max_iter, int) or max_iter <= 0):
            raise ValueError('Maximum number of iterations should be a positive integer.')
        if max_time is not None and (not isinstance(max_time, int) or max_time <= 0):
            raise ValueError('Maximum amount of time should be a positive integer.')
        if not isinstance(verbose, int) or verbose < -1 or verbose > 2:
            raise ValueError('Verbose should be an integer between -1 and 2.')
        info = {'k': k,
                'alg': str(self.__class__),
                'A_dim_1': A.shape[0],
                'A_dim_2': A.shape[1],
                'A_type': str(A.__class__),
                'max_iter': max_iter if max_iter is not None else self.default_max_iter,
                'verbose': verbose,
                'max_time': max_time if max_time is not None else self.default_max_time}
        if init is not None:
            W = init[0].copy()
            H = init[1].copy()
            info['init'] = 'user_provided'
        else:
            W = np.random.rand(A.shape[0], k)
            H = np.random.rand(A.shape[1], k)
            info['init'] = 'uniform_random'

        if verbose >= 0:
            print('[DataSupervisionNMF] Running: ')
            print(json.dumps(info, indent=4, sort_keys=True))

        norm_A = mu.norm_fro(A)
        total_time = 0

        if verbose >= 1:
            his = {'iter': [], 'elapsed': [], 'rel_error': []}

        start = time.time()
        # algorithm-specific initilization
        (W, H) = self.initializer(W, H)

        for i in range(1, info['max_iter'] + 1):
            start_iter = time.time()
            # algorithm-specific iteration solver
            (W, H), err = self.iter_solver(A, W, H, k, i, 
                feature_split_points, data_split_points, normA=norm_A)
            elapsed = time.time() - start_iter

            if verbose >= 1:
                rel_error = err / norm_A
                his['iter'].append(i)
                his['elapsed'].append(elapsed)
                his['rel_error'].append(rel_error)
                if verbose >= 2:
                    print('iter: {0}, elapsed: {1:.4f}, rel_error: {2:.4f}'.format(i, elapsed, rel_error))

            total_time += elapsed
            if total_time > info['max_time']:
                break

        W, H, _ = mu.normalize_column_pair(W, H)

        final = {}
        final['norm_A'] = norm_A
        final['rel_error'] = err / norm_A
        final['iterations'] = i
        final['elapsed'] = time.time() - start

        rec = {'info': info, 'final': final}
        if verbose >= 1:
            rec['his'] = his

        if verbose >= 0:
            print('[DataSupervisionNMF] Completed: ')
            print(json.dumps(final, indent=4, sort_keys=True))
        return W, H, rec

    def run_repeat(self, A, k, num_trial, max_iter=None, max_time=None, verbose=0):
        """ Run an NMF algorithm several times with random initial values 
            and return the best result in terms of the Frobenius norm of
            the approximation error matrix

        Parameters
        ----------
        A : numpy.array or scipy.sparse matrix, shape (m,n)
        k : int - target lower rank
        num_trial : int number of trials

        Optional Parameters
        -------------------
        max_iter : int - maximum number of iterations for each trial.
                    If not provided, default maximum for each algorithm is used.
        max_time : int - maximum amount of time in seconds for each trial.
                    If not provided, default maximum for each algorithm is used.
        verbose : int - 0 (default) - No debugging information is collected, but
                                    input and output information is printed on screen.
                        -1 - No debugging information is collected, and
                                    nothing is printed on screen.
                        1 (debugging/experimental purpose) - History of computation is
                                        returned. See 'rec' variable.
                        2 (debugging/experimental purpose) - History of computation is
                                        additionally printed on screen.
        Returns
        -------
        (W, H, rec)
        W : Obtained factor matrix, shape (m,k)
        H : Obtained coefficient matrix, shape (n,k)
        rec : dict - (debugging/experimental purpose) Auxiliary information about the execution
        """

        best = None
        best_error = np.inf
        for t in range(num_trial):
            if verbose >= 0:
                print('[NMF] Running the {0}/{1}-th trial ...'.format(t + 1, num_trial))
            this = self.run(A, k, verbose=(-1 if verbose == 0 else verbose))
            error = this[2]['final']['rel_error']
            if error < best_error:
                best_error = error
                best = this
        if verbose >= 0:
            print('[NMF] Best result is as follows.')
            print(json.dumps(best[2]['final'], indent=4, sort_keys=True))
        return best[0], best[1], best[2]

    def iter_solver(self, A, W, H, k, it, 
            feature_split_points = [], data_split_points = [], normA=0):
        Sol, err = nnlsm_blockpivot_data_supervised(H, A.T, init=W.T, for_H=False,
                feature_split_points=feature_split_points, data_split_points=data_split_points,
                normA=normA)
        W = Sol.T
        Sol, _ = nnlsm_blockpivot_data_supervised(W, A, init=H.T, for_H=True,
                feature_split_points=feature_split_points, data_split_points=data_split_points)
        H = Sol.T
        return (W, H), err

    def initializer(self, W, H):
        return (W, H)