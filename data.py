#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

data = pd.read_csv('dye_data.csv', encoding='utf-8')

_keys = data.columns

A = data[[key for key in _keys if key.startswith('D')]].values
A = np.array([a / a.mean() for a in A])
B = data[[key for key in _keys if key.startswith('LS')]].values