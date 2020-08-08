# -*- coding: utf-8 -*-
"""
Created on 2020/2/4
Python 3.6

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


import os
import time
import pandas as pd
import numpy as np
import networkx as nx
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import itertools
import warnings

warnings.filterwarnings('ignore')

data_path = 'data/'
save_path = 'output/'

if os.path.exists(save_path):
    pass
else:
    os.makedirs(save_path)
