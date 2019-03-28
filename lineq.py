#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Error Curve

solve the eq, then draw the error curve
"""


import numpy as np
import numpy.linalg as LA
from sklearn.model_selection import train_test_split

from utils import *
from data import *

if __name__ == '__main__':

    W, s, Wh = LA.svd(B.T @ B)

    A, A_test, B, B_test = train_test_split(A, B, test_size=0.2)

    Es1 = []
    Es2 = []

    N, r = A.shape
    V, s, Vh = LA.svd(A.T @ A)
    C = A @ V
    q = 4

    s /= np.sum(s)
    ss = np.cumsum(s)
    ps = np.arange(1, 70, 5)

    for p in ps:
        
        B1 = B @ W[:, :q]
        B1_test = B_test @ W[:,:q]
        C1 = C[:,:p]

        Y = LA.lstsq(C1, B1, rcond=None)[0]
        XX = V[:,:p] @ Y
        XX = max0(XX @ Wh[:q, :]) @ W[:, :q]

        E1 = relerror(A @ XX, B1)
        E2 = relerror(A_test @ XX, B1_test)

        Es1.append(E1)
        Es2.append(E2)
 
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    myfont = FontProperties(fname='/System/Library/Fonts/PingFang.ttc')

    ax = plt.subplot(111)
    ax.set_xlabel("A 主成分数", fontproperties=myfont)
    ax.set_ylabel("相对误差", fontproperties=myfont)

    ax.set_title('主成分数-误差关系图', fontproperties=myfont)
    ax.plot(ps, Es1, '-o', ps, Es2, '-s')

    ax.legend(('降维方程组误差', '预测误差'), prop=myfont)

    XX = A @ LA.lstsq(A, B, rcond=None)[0] - B
    re = error(XX) / error(B)
    ax.plot((ps[0], ps[-1]), [re, re], '--k')
    ax.annotate('原方程相对误差', xy = (ps[0], re), xytext=(ps[0], re + 0.1), arrowprops={'arrowstyle':'->'}, fontproperties=myfont)

    ret = error(A_test @ LA.lstsq(A_test, B_test, rcond=None)[0], B_test) / error(B_test)
    ax.plot((ps[0], ps[-1]), [ret, ret], '--g')
    ax.annotate('预测相对误差', color='green', xy=(ps[0], ret), xytext=(ps[0], ret - 0.1), arrowprops={'arrowstyle':'->', 'color':'green'}, fontproperties=myfont)

    tax = ax.twinx()
    tax.plot(ps, ss[:69:5], color='m')
    tax.set_ylabel('累计百分比', fontproperties=myfont)
    tax.legend(('累计百分比',), prop=myfont)
 
    plt.show()
