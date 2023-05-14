# -*- coding: utf-8 -*-
"""
Created on Mon May  8 19:02:10 2023

@author: pf259
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import pycodamath as coda
from pycodamath import plot
import numpy as np
from functions import get_entity_dict, plot_entity_data, contrast_matrix, OLS, plot_ternary
from sklearn.cluster import KMeans
import copy

# %% Load raw data
# Get the directory path of the current script file
dir_path = os.path.dirname(os.path.realpath(__file__))

# Set the working directory to the script directory
os.chdir(dir_path)

# Set raw data location
raw_data_location = os.path.join(os.pardir, 'data', 'World_Energy_sources.csv')

raw_data = pd.read_csv(raw_data_location, delimiter=';')

# There is no obvious reason to treat NaN and 0 differently
# (NaN appear only in Bioenergy and Other)
# so let's substitute NaN with 0
raw_data.fillna(0, inplace=True)

# Split dataset in different countries
data = {} # Store data tables here
data['composition'] = get_entity_dict(raw_data)

# %% Handle zeros

# Oceania never produced nuclear, so let's drop that column
# (leave out parts with structural zeros)
data['composition']['Oceania'] = data['composition']['Oceania'].drop('Nuclear', axis=1)

# The composition has 9 parts, so when we have 0 in one part it is more than 
# 10% already → parametric replacement
for key in data['composition']:
    data['composition'][key] = data['composition'][key].coda.zero_replacement(n_samples=5000)
del(key)
    
# It seems like coda.zero_replacement closes the composition to 1

# %% Closure to 100%
data['percent'] = {}
for key in data['composition']:
    data['percent'][key] = data['composition'][key].coda.closure(100)

# Define color set
colors_set = plt.cm.get_cmap('Set1')
colors_dict = {}
for i, col in enumerate( data['composition']['Africa']):
    colors_dict[col] = colors_set(i)
colors_dict['Wind'] = '#2eb8b8' #Change because yellow can't be seen

# Plot
plot_paths = {} # Store paths to plots here
plot_paths['percent'] = os.path.join(os.pardir, 'plots', '01_percent')
plot_entity_data(data['percent'], colors_dict, plot_paths['percent'], y_label='% Power production')

# %% PCA biplot

# The Biplot function centers, scales and applies CLR to data
# It is enough to feed it the closed data
plot_paths['PCA'] = os.path.join(os.pardir, 'plots', '02_PCA')
for key in data['percent']:
    mypca = coda.pca.Biplot(data['percent'][key])
    mypca.plotscorelabels()
    plt.title(key)
    # Save file
    if not os.path.exists(plot_paths['PCA']):
        os.makedirs(plot_paths['PCA'])
    file = os.path.join(plot_paths['PCA'], key)
    plt.savefig(file, dpi=300,  bbox_inches='tight')
    plt.close()
    
# The PCAs show the following correlations
# Classic: Coal, Gas, Oil, Hydro, Nuclear
# Modern: Solar, Wind
# Alternative: Bioenergy, Other
# I will use these partitions.
# Modern and alternative are always simmetric to Classic, so for the ILR
# I should first split this way

# In every single graph the years until 1999 (included) and after 2000
# form two well distinguished clusters. I will split them in two ternary plot.
# In each plot, I will color the various countries with different colors,
# hhoping to see nice clusters and draw conclusions on energy usage

# %% Linear regression 

# Partition table and contrast matrix (psi)
# NB: Oceania does not consider Nuclear
partition_table = np.array([[1, 1, 1, 1, 1, -1, -1, -1, -1], 
                            [1, -1, -1, 1, 1, 0, 0, 0, 0], 
                            [0, 1, -1, 0, 0, 0, 0, 0, 0], 
                            [1, 0, 0, 1, -1, 0, 0, 0, 0], 
                            [1, 0, 0, -1, 0, 0, 0, 0, 0], 
                            [0, 0, 0, 0, 0, 1, 1, -1, -1], 
                            [0, 0, 0, 0, 0, 1, -1, 0, 0], 
                            [0, 0, 0, 0, 0, 0, 0, 1, -1]])

ILR_coordinates = ['classic/others', 
                   '(coal, nuclear, hydro)/(gas, oil)',
                   'gas/oil',
                   '(coal, nuclear)/hydro',
                   'coal/nuclear',
                   '(wind, solar)/(bioenergy, other)',
                   'wind/solar',
                   'bioenergy/other']
ILR_coordinates_oceania = ['classic/others',
                           '(coal, hydro)/(gas, oil)',
                           'gas/oil',
                           'coal/hydro',
                           '(wind, solar)/(bioenergy, other)',
                           'wind/solar',
                           'bioenergy/other']

partition_table_oceania = copy.deepcopy(partition_table)
partition_table_oceania = np.delete(partition_table_oceania, 3, axis=1)
partition_table_oceania = np.delete(partition_table_oceania, 4, axis=0)
psi = contrast_matrix(partition_table)
psi_oceania = contrast_matrix(partition_table_oceania)

# ILR
data['ILR'] = {}
for key in data['percent'].keys() - {'Oceania'}:
    data['ILR'][key] = data['percent'][key].coda.ilr(psi)
    data['ILR'][key].columns = ILR_coordinates
data['ILR']['Oceania'] = data['percent']['Oceania'].coda.ilr(psi_oceania)
data['ILR']['Oceania'].columns = ILR_coordinates_oceania

# Plot
for i, col in enumerate(data['ILR']['Africa']):
    colors_dict[col] = colors_set(i)
colors_dict['(coal, hydro)/(gas, oil)'] = colors_dict['(coal, nuclear, hydro)/(gas, oil)'] 
colors_dict['coal/hydro'] = colors_dict['(coal, nuclear)/hydro'] 
colors_dict['(wind, solar)/(bioenergy, other)'] = '#2eb8b8' #Change because yellow can't be seen

plot_paths['ILR'] = os.path.join(os.pardir, 'plots', '03_ILR')
plot_entity_data(data['ILR'], colors_dict, plot_paths['ILR'], y_label='ILR coordinates')

# Most of the data changes trajectory after the year 2000
# So I will fit the model from that date on
prediction_range = range(2000, 2031)

data['linear_ILR'] = {}
for key in data['ILR']:
    data['linear_ILR'][key] = OLS(data['ILR'][key], prediction_range)

# Plot
plot_paths['linear_ILR'] = os.path.join(os.pardir, 'plots', '04_linear_ILR')
plot_entity_data(data['ILR'], colors_dict, plot_paths['linear_ILR'], y_label='ILR coordinates', pred=data['linear_ILR'])

# Inverse ILR (back to R space)
# NB: pycodamath function does not apply closure, so I'll do it
data['linear'] = {}
for key in data['linear_ILR'].keys() - {'Oceania'}:
    data['linear'][key] = data['linear_ILR'][key].coda.ilr_inv(psi)
    data['linear'][key] = data['linear'][key].coda.closure(100)
    data['linear'][key].index = prediction_range
data['linear']['Oceania'] = data['linear_ILR']['Oceania'].coda.ilr_inv(psi_oceania)
data['linear']['Oceania'] = data['linear']['Oceania'].coda.closure(100)
data['linear']['Oceania'].index = prediction_range

# Plot
plot_paths['linear'] = os.path.join(os.pardir, 'plots', '05_linear')
plot_entity_data(data['percent'], colors_dict, plot_paths['linear'], y_label='% Power production', pred=data['linear'])

# In many countries there are one or two sources that are mostly used 
# usually coal and gas (hydro for SOuth America)
# I suspect that it would be useful to perform linear regression on a 
# subcomposition which excludes them

# %% Ternary plot

# Amalgamation according to PCA correlations
data['ternary'] = {}
for key in data['percent'].keys() - {'Oceania'}:
    data['ternary'][key] = pd.DataFrame()
    data['ternary'][key]['Classic'] = data['percent'][key]['Coal'] + data['percent'][key]['Gas'] + data['percent'][key]['Oil'] + data['percent'][key]['Hydro'] + data['percent'][key]['Nuclear']
    data['ternary'][key]['Modern'] = data['percent'][key]['Solar'] + data['percent'][key]['Wind']
    data['ternary'][key]['Alternative'] = data['percent'][key]['Bioenergy'] + data['percent'][key]['Other']
data['ternary']['Oceania'] = pd.DataFrame()
data['ternary']['Oceania']['Classic'] = data['percent']['Oceania']['Coal'] + data['percent']['Oceania']['Gas'] + data['percent']['Oceania']['Oil'] + data['percent']['Oceania']['Hydro']
data['ternary']['Oceania']['Modern'] = data['percent']['Oceania']['Solar'] + data['percent']['Oceania']['Wind']
data['ternary']['Oceania']['Alternative'] = data['percent']['Oceania']['Bioenergy'] + data['percent']['Oceania']['Other']

# Ternary data originates from percent data.
# Percent data's zeros have been replaced already, and it is closed to 100
# So there is no need to do it again

# Split on year 2000
data['ternary_1900'] = {}
data['ternary_2000'] = {}
for key in data['ternary']:
    data['ternary_1900'][key] = data['ternary'][key].loc[: 1999]
    data['ternary_2000'][key] = data['ternary'][key].loc[2000 :]

# Plot
# NB: the geometric mean is calculated on the whole dataset
# not for each country
for i, col in enumerate(data['ternary']):
    colors_dict[col] = colors_set(i)
colors_dict['Oceania'] = '#2eb8b8' #Change because yellow can't be seen

plot_paths['ternary'] = os.path.join(os.pardir, 'plots', '06_ternary')
plot_ternary(data['ternary'], colors_dict, center=False, destination_folder=plot_paths['ternary'], name='all') # All year, not center
plot_ternary(data['ternary'], colors_dict, center=True, destination_folder=plot_paths['ternary'], name='all_center') # All year, not center
plot_ternary(data['ternary_1900'], colors_dict, center=True, destination_folder=plot_paths['ternary'], name='1900') # Center
plot_ternary(data['ternary_2000'], colors_dict, center=True, destination_folder=plot_paths['ternary'], name='2000') # Center
# TODO: ILR coordinates?

# %% Delete temp variables because of OCD
del (col, file, i, key, mypca)
