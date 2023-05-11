# -*- coding: utf-8 -*-
"""
Created on Thu May 11 21:47:25 2023

@author: acer
"""

''' Clustering and Fitting'''

from sklearn.metrics import silhouette_score
from sklearn import cluster
import numpy as np
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
%matplotlib inline

def Expo(t, scale, growth):
    f = (scale * np.exp(growth * (t-1960)))
    return f


def func(x, k, l, m):
    """Function to use for finding error ranges"""
    k, x, l, m = 0, 0, 0, 0
    return k * np.exp(-(x-l)**2/m)


def err_ranges(x, func, param, sigma):
    """Function to find error ranges for fitted data
    x: x array of the data
    func: defined function above
    param: parameter found by fitted data
    sigma: sigma found by fitted data"""
    import itertools as iter

    low = func(x, *param)
    up = low

    uplow = []
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        low = np.minimum(low, y)
        up = np.maximum(up, y)

    return low, up


def read_data(filename):
    '''
    reading excel file
    '''
    df = pd.read_excel(filename, skiprows=3)
    return df


def stat_data(df, col, value, yr, a):
    '''
    code for filtering data
    '''
    df3 = df.groupby(col, group_keys=True)
    df3 = df3.get_group(value)
    df3 = df3.reset_index()
    #setting index
    df3.set_index('Indicator Name', inplace=True)
    df3 = df3.loc[:, yr]
    #transposing
    df3 = df3.transpose()
    df3 = df3.loc[:, a]
    df3 = df3.dropna(axis=1)
    return df3


#reading dataset
env_data = read_data("API_6_DS2_en_excel_v2_5361655 (1).xlsx")