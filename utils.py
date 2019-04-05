#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 

import numpy as np
import numpy.linalg as LA
from scipy.optimize import lsq_linear

from sklearn.linear_model import Lasso

def max_(x):
    return max(x, 0)

def error(A, B=None):
    if B is None:
        return LA.norm(A, 'f')
    else:
        return LA.norm((A-B), 'f')

def relerror(A, B):
    return LA.norm((A-B), 'f') / LA.norm(B, 'f')

def max0(x):
    return np.frompyfunc(max_, 1, 1)(x).astype('float64')

def cut_(x, eps):
    return 0 if x < eps else x

def cut(x, eps):
    return np.frompyfunc(lambda x: cut_(x, eps), 1, 1)(x)

def lsq(A, B):
    xs = [lsq_linear(A, B[:, k])['x'] for k in range(B.shape[1])]
    return np.column_stack(xs)

def solve_pca(A, B, p=3, q=None):
    """Solve AX=B with PCA
    
    Arguments:
        A {np.ndarray} -- Matrix
        B {np.ndarray} -- Matrix
    
    Keyword Arguments:
        p {number} -- number of components (default: {3})
        q {number} -- number of components (default: {None})
    
    Returns:
        np.ndarray -- solution
    """

    V, _, Vh = LA.svd(A.T @ A)
    Vp = V[:, :p]
    Cp = A @ Vp
    if q:
        W, _, Wh = LA.svd(B.T @ B)
        Wq = W[:, :q]
        Bq = B @ Wq
        Y = LA.lstsq(Cp, Bq, rcond=None)[0]
        X = Vp @ Y @ Wh[:q,:]
    else:
        Y = LA.lstsq(Cp, B, rcond=None)[0]
        X = Vp @ Y
    # Err = relerror(A @ X, B)
    return X, 0

solve = solve_pca

from sklearn.decomposition import NMF
def solve_nmf(A, B, p=3, q=None):
    nmf = NMF(n_components=p, init='nndsvd')
    nmf.fit(A)
    W = nmf.transform(A)
    H = nmf.components_
    if q:
        U, _, Uh = LA.svd(B.T @ B)
        Uq = U[:, :q]
        Bq = B @ Uq
        Y = LA.lstsq(W, Bq, rcond=None)[0]
        X = LA.lstsq(H, Y, rcond=None)[0] @ Uh[:q,:]
    else:
        Y = LA.lstsq(U, B, rcond=None)[0]
        X = LA.lstsq(H, Y, rcond=None)[0]
    return X, 0
