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

#selecting year
start = 1960
end = 2015
year = [str(i) for i in range(start, end+1)]
#selecting indicator for fitting
Indicator = [
    'CO2 emissions from residential buildings and commercial and public services (% of total fuel combustion)', 'CO2 emissions from solid fuel consumption (% of total)']
data = stat_data(env_data, 'Country Name', 'India', year, Indicator)
#selecting indicator for clustering
Indicator1 = ['Electricity production from renewable sources, excluding hydroelectric (% of total)', 'CO2 emissions from solid fuel consumption (% of total)', 'Electricity production from oil, gas and coal sources (% of total)',
              'CO2 emissions from liquid fuel consumption (% of total)', 'CO2 emissions from other sectors, excluding residential buildings and commercial and public services (% of total fuel combustion)']
data1 = stat_data(env_data, 'Country Name', 'Australia', year, Indicator1)
#renaming data
data = data.rename_axis('Year').reset_index()
data['Year'] = data['Year'].astype('int')
data.dtypes
#shortening indicators
data1 = data1.rename(columns={
    'Electricity production from renewable sources, excluding hydroelectric (% of total)': 'Eletricity production(renewable)',
    'CO2 emissions from solid fuel consumption (% of total)': 'CO2 emission(solid)',
    'Electricity production from oil, gas and coal sources (% of total)': 'electricity production (total)',
    'CO2 emissions from liquid fuel consumption (% of total)': 'CO2 emission(liquid)',
    'CO2 emissions from other sectors, excluding residential buildings and commercial and public services (% of total fuel combustion)': 'CO2 emission(other sector)'})


def map_corr(df, size=6):
    """Function creates heatmap of correlation matrix for each pair of 
    columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot (in inch)
        
    """
    corr = df.corr()
    plt.figure(figsize=(size, size))
    # fig, ax = plt.subplots()
    plt.matshow(corr, cmap='coolwarm')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns,
               rotation=90, color='maroon')
    plt.yticks(range(len(corr.columns)), corr.columns, color='maroon')
    plt.title("Heatmap of correlation matrix", color='red')
    plt.colorbar()
    plt.savefig("Heatmap of correlation matrix.png")


#correlation of datas
corr = data1.corr()
map_corr(data1)
plt.show()

#scatter matrix
plt.figure()
pd.plotting.scatter_matrix(data1, figsize=(9, 9))
plt.tight_layout()
#setting the title
plt.title("Scater matrix")
plt.savefig("scatter matrix.png")

#scatterplot before fitting
plt.figure()
plt.scatter(
    data["Year"], data["CO2 emissions from solid fuel consumption (% of total)"])
plt.title('Scatter Plot between 1960-2010 before fitting', color='red')
plt.ylabel('CO2 emissions from solid fuel consumption (% of total)')
#setting the x limit
plt.xlim(1960, 2015)
plt.savefig("Scatter_fit.png")
plt.show()

#plot after fitting
popt, pcov = opt.curve_fit(
    Expo, data['Year'], data['CO2 emissions from solid fuel consumption (% of total)'], p0=[1000, 0.02])
data["Pop"] = Expo(data['Year'], *popt)
sigma = np.sqrt(np.diag(pcov))
low, up = err_ranges(data["Year"], Expo, popt, sigma)
plt.figure()
plt.title("Plot After Fitting")
plt.plot(data["Year"],
         data['CO2 emissions from solid fuel consumption (% of total)'], label="data")
plt.plot(data["Year"], data["Pop"], label="fit")
plt.fill_between(data["Year"], low, up, alpha=0.7)
plt.legend()
plt.show()
plt.savefig("Plot After Fitting.png")

#Predicting future values
low, up = err_ranges(2030, Expo, popt, sigma)
print("CO2 emissions from solid fuel consumption in 2030 is ", low, "and", up)
low, up = err_ranges(2040, Expo, popt, sigma)
print("CO2 emissions from solid fuel consumption in 2040 is ", low, "and", up)

scaler = MinMaxScaler()
scaler.fit(data1[['CO2 emission(solid)']])
data['Scaler_T'] = scaler.transform(
    data1['CO2 emission(solid)'].values.reshape(-1, 1))

scaler.fit(data1[['CO2 emission(liquid)']])
data['Scaler_M'] = scaler.transform(
    data1['CO2 emission(liquid)'].values.reshape(-1, 1))
data_c = data.loc[:, ['Scaler_T', 'Scaler_M']]
print(data_c)


def n_cluster(data_frame):
  k_rng = range(1, 10)
  sse = []
  for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit_predict(data_frame)
    sse.append(km.inertia_)
  return k_rng, sse


a, b = n_cluster(data_c)
#setting x and y label
plt.xlabel = ('k')
plt.ylabel('sum of squared error')
plt.plot(a, b)
plt.title("Number of clusters of Co2 emission of soild fuel", color='red')
print(b)
plt.savefig("Number of clusters.png")

#code clustering
km = KMeans(n_clusters=2)
pred = km.fit_predict(data_c[['Scaler_T', 'Scaler_M']])
data_c['cludter'] = pred
data_c.head()

centers = km.cluster_centers_

dc1 = data_c[data_c.cludter == 0]
dc2 = data_c[data_c.cludter == 1]
plt.figure()
plt.scatter(dc1['Scaler_T'], dc1['Scaler_M'], color='green')
plt.scatter(dc2['Scaler_T'], dc2['Scaler_M'], color='red')
plt.scatter(centers[:, 0], centers[:, 1], s=200, marker='*', color='black')
plt.legend()
plt.title("Clustering", color='red')
plt.show()
plt.savefig("Cluster.png")
