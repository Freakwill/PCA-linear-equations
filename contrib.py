#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 

"""Contributes
Calculate the contributes, then draw them as a curve.
"""


import numpy as np
import numpy.linalg as LA
import pandas as pd
from sklearn.model_selection import train_test_split

from data import *

V, s, Vh = LA.svd(A.T @ A)
W, sb, Wh = LA.svd(B.T @ B)

s /= np.sum(s)
ss = np.cumsum(s)
ps = np.arange(70)
contribs=ss[:70]

sb /= np.sum(sb)
ssb = np.cumsum(sb)
psb = np.arange(30)
contribsb=ssb[:30]

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
myfont = FontProperties(fname='/System/Library/Fonts/PingFang.ttc')

ax = plt.subplot(121)
ax.set_xlabel("A 主成分数", fontproperties=myfont)
ax.set_ylabel("特征值累计百分比", fontproperties=myfont)

ax.plot(ps+1, contribs, 'b-.')
n = 30
ax.annotate('推荐主成分数(%d)' % n, color='red', xy=(n, ss[n-1]), xytext=(n, ss[n-1] - 0.2), arrowprops={'arrowstyle':'->', 'color':'blue'}, fontproperties=myfont)
ax.legend(('A矩阵PCA的累计百分比',), prop=myfont)

ax = plt.subplot(122)
ax.set_xlabel("B 主成分数", fontproperties=myfont)

ax.plot(psb+1, contribsb, 'g-.')
q = 4
ax.annotate('推荐主成分数(%d)' % q, color='red', xy=(q, ssb[q-1]), xytext=(q, ssb[q-1] - 0.02), arrowprops={'arrowstyle':'->', 'color':'green'}, fontproperties=myfont)
ax.legend(('B矩阵PCA的累计百分比',), prop=myfont)
plt.suptitle('主成分数-累计百分比关系图', fontproperties=myfont)
plt.show()
