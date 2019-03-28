#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Error Curve

solve the eq, then draw the error curve
"""

import time
import numpy as np
import numpy.linalg as LA
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import *
from data import *

def perf(A, B, A_test, B_test, ps, q=4, method='pca'):

    Errs = []
    Errs_test = []
    times = []
    for p in ps:
        time1 = time.perf_counter()
        B1 = B @ W[:, :q]

        XX, _ = globals()['solve_%s'%method](A, B1, p)
        #XX = max0(XX @ Wh[:q, :]) @ W[:, :q]
        Err = relerror(A @ XX @ Wh[:q,:], B)

        time2 = time.perf_counter() - time1
        times.append(time2)

        Err_test = relerror(A_test @ XX @ Wh[:q,:], B_test)

        Errs.append(Err)
        Errs_test.append(Err_test)

    globals()[method] = {'time': np.array(times), 'error':Errs, 'test':Errs_test}



if __name__ == '__main__':

    W, s, Wh = LA.svd(B.T @ B)

    A, A_test, B, B_test = train_test_split(A, B, test_size=0.2)

    ps = np.arange(1, 30)

    perf(A, B, A_test, B_test, ps)
    perf(A, B, A_test, B_test, ps, method='nmf')

    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    myfont = FontProperties(fname='/System/Library/Fonts/PingFang.ttc')

    ax = plt.subplot(111)
    ax.set_xlabel("A 主成分数", fontproperties=myfont)
    ax.set_ylabel("相对误差", fontproperties=myfont)
    ax.set_title('PCA, NMF预测误差与相对用时比较', fontproperties=myfont)

    ax.plot(ps, pca['test'], '-o')
    ax.plot(ps, nmf['test'], '-d')
    ax.legend(('PCA预测误差', 'NMF预测误差'), prop=myfont)

    time1 = time.perf_counter()
    XX = LA.lstsq(A, B, rcond=None)[0]
    re = relerror(A @ XX, B)
    time2 = time.perf_counter() - time1

    # ax.plot((ps[0], ps[-1]), [re, re], '--k')
    # ax.annotate('原方程相对误差', xy = (ps[0], re), xytext=(ps[0], re + 0.05), arrowprops={'arrowstyle':'->'}, fontproperties=myfont)

    # XXt = LA.lstsq(A_test, B_test, rcond=None)[0]
    # ret = relerror(A_test @ XXt, B_test)
    # ax.plot((ps[0], ps[-1]), [ret, ret], '--g')
    # ax.annotate('预测相对误差', color='green', xy=(ps[0], ret), xytext=(ps[0], ret - 0.05), arrowprops={'arrowstyle':'->', 'color':'green'}, fontproperties=myfont)

    tax = ax.twinx()
    tax.plot(ps, pca['time']/time2, '-.', ps, nmf['time']/time2, '-.+')
    tax.set_ylabel('降维用时/不降维用时', fontproperties=myfont)
    tax.legend(('PCA相对用时', 'NMF相对用时'), prop=myfont)
 
    plt.show()
