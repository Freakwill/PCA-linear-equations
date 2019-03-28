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

def solve_pca(A, B, p=3):
    """Solve AX=B with PCA
    
    Arguments:
        A {np.ndarray} -- Matrix
        B {np.ndarray} -- Matrix
    
    Keyword Arguments:
        p {number} -- number of components (default: {3})
    
    Returns:
        np.ndarray -- solution
    """

    V, s, Vh = LA.svd(A.T @ A)
    Vp = V[:, :p]
    Cp = A @ Vp
    Y = LA.lstsq(Cp, B, rcond=None)[0]
    X = Vp @ Y
    # Err = relerror(A @ X, B)
    return X, Err

solve = solve_pca

from sklearn.decomposition import NMF
def solve_nmf(A, B, p=3):
    nmf = NMF(n_components=p)
    nmf.fit(A)
    W = nmf.transform(A)
    H = nmf.components_
    Y = LA.lstsq(W, B, rcond=None)[0]
    X = lsq(H, Y)
    # Err = relerror(A @ X, B)
    return X, Err
