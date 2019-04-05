#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
import time

# import logging
# logging.basicConfig(level=logging.ERROR)

import numpy as np
import numpy.linalg as LA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn.neural_network import MLPRegressor

from utils import *
from data import *

A, A_test, B, B_test = train_test_split(A, B, test_size=0.2)

V, s, Vh = LA.svd(A.T @ A)
W, s, Wh = LA.svd(B.T @ B)

p, q = 30, 4

Vp = V[:, :p]
Wq = W[:,:q]
Cp = A @ Vp
Bq = B @ Wq

models = (
# 'ARDRegression',
'BayesianRidge',
'ElasticNet',
'ElasticNetCV',
'HuberRegressor',
'Lars',
'LarsCV',
'Lasso',
'LassoCV',
'LassoLars',
'LassoLarsCV',
'MLPRegressor'
)


with open('error.txt', 'a') as f:
    for model_name in models:
        model = locals()[model_name]()
        errors = []
        times = []
        for _ in range(2):
            time1 = time.perf_counter()
            Bs = []
            for k in range(q):
                model.fit(Cp, Bq[:,k])
                Bs.append(model.predict(A_test @ Vp))

            Bs = np.column_stack(Bs)
            error = relerror(Bs @ Wh[:q, :], B_test)
            errors.append(error)
            time2 = time.perf_counter() - time1
            times.append(time2)
        
        f.write(f"{model_name}测试误差: {np.mean(errors):.4f} ({np.mean(times):.4f}sec)\n")