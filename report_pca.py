#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import numpy.linalg as LA
from sklearn.model_selection import train_test_split

import time

from utils import *
from data import *


W, s, Wh = LA.svd(B.T @ B)
A, A_test, B, B_test = train_test_split(A, B, test_size=0.2)

N, r = A.shape
N, c = B.shape

time1 = time.perf_counter()
p, q =30, 4

B1 = B @ W[:, :q]
B1_test = B_test @ W[:,:q]

XX, _ = solve(A, B1, p)
XX = max0(XX @ Wh[:q, :]) @ W[:, :q]
Err = relerror(A @ XX @ Wh[:q,:], B)
delta_time = time.perf_counter() - time1

Err_test = relerror(A_test @ XX @ Wh[:q,:], B_test)


print(f'''
    解 AX=B
    A: {N} X {r}
    B: {N} X {c}
    ------------
    A 主成分数: {p}
    B 主成分数: {q}
    降维训练误差: {Err:.4f} (用时 {delta_time:.4f}sec)
    测试误差: {Err_test:.4f}''')

X = LA.lstsq(A, B, rcond=None)[0]
print(f'原方程误差: {relerror(A @ X, B):.4f}')
