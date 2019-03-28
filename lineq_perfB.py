#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Error Curve related to components of B

solve the eq, then draw the error curve

The number of components of A is given.
"""

import time
import numpy as np
import numpy.linalg as LA
from sklearn.model_selection import train_test_split

from utils import *
from data import *

if __name__ == '__main__':
    

    rb = B.shape[1]

    W, s, Wh = LA.svd(B.T @ B)

    A, A_test, B, B_test = train_test_split(A, B, test_size=0.2)

    time1 = time.perf_counter()
    N, r = A.shape

    XX = LA.lstsq(A, B, rcond=None)[0]
    time2 = time.perf_counter() - time1

    XXt = LA.lstsq(A_test, B_test, rcond=None)[0]
    ret = relerror(A_test @ XXt, B_test)

    re = relerror(A @ XX, B)
    ret = relerror(A_test @ XXt, B_test)

    Es1 = []
    Es2 = []
    
    # s /= np.sum(s)
    # ss = np.cumsum(s)

    qs = np.arange(1, 35, 2)
    times = []
    for q in qs:
        time1 = time.perf_counter()
        p = 30
        B1 = B @ W[:, :q]
        B1_test = B_test @ W[:,:q]

        N, r = A.shape
        XX, _ = solve(A, B1, p)
        # XX = max0(XX @ Wh[:q, :]) @ W[:, :q]

        time2 = time.perf_counter() - time1
        times.append(time2)

        XXt, _ = solve(A_test, B1_test, p)
        XXt = max0(XXt @ Wh[:q, :]) @ W[:, :q]

        E1 = relerror(A @ np.hstack([XX, np.zeros((XX.shape[0], rb-q))]) @ W.T, B)
        E2 = relerror(A_test @ np.hstack([XXt, np.zeros((XXt.shape[0], rb-q))]) @ W.T, B_test)

        Es1.append(E1)
        Es2.append(E2)
 
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    myfont = FontProperties(fname='/System/Library/Fonts/PingFang.ttc')

    ax = plt.subplot(111)
    ax.set_xlabel("B 主成分数", fontproperties=myfont)
    ax.set_ylabel("相对误差", fontproperties=myfont)

    ax.set_title('主成分数-误差关系图', fontproperties=myfont)
    ax.plot(qs, Es1, '-o', qs, Es2, '-s')

    ax.legend(('降维方程组误差', '预测误差'), prop=myfont)
    
    ax.plot((qs[0], qs[-1]), [re, re], '--k')
    ax.annotate('原方程相对误差', xy = (qs[0], re), xytext=(qs[0], re + 0.01), arrowprops={'arrowstyle':'->'}, fontproperties=myfont)

    ax.plot((qs[0], qs[-1]), [ret, ret], '--g')
    ax.annotate('预测相对误差', color='green', xy=(qs[0], ret), xytext=(qs[0], ret - 0.01), arrowprops={'arrowstyle':'->', 'color':'green'}, fontproperties=myfont)


    tax = ax.twinx()
    tax.plot(qs, np.array(times)/time2, '-.')
    tax.set_ylabel('降维用时/不降维用时', fontproperties=myfont)
    tax.legend(('相对用时',), prop=myfont)
 
    plt.show()
