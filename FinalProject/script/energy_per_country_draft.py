# -*- coding: utf-8 -*-
"""
Created on Sat May  6 16:14:12 2023

@author: pf259
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import pycodamath as coda
from pycodamath import plot
import numpy as np
from functions import get_entity_dict, plot_entity_data, contrast_matrix, OLS

# %% Load raw data
# Get the directory path of the current script file
dir_path = os.path.dirname(os.path.realpath(__file__))

# Set the working directory to the script directory
os.chdir(dir_path)

# Set raw data location
raw_data_location = os.path.join(os.pardir, 'data', 'World_Energy_sources.csv')

# Read raw data
raw_data = pd.read_csv(raw_data_location, delimiter=';')

# There is no obvious reason to treat NaN and 0 differently
# (they appear only in Bioenergy and Other)
# so let's substitute NaN with 0
raw_data.fillna(0, inplace=True)

# Split dataset in different countries
entity_tables = get_entity_dict(raw_data)


# %% Handle zeros

# Oceania never produced nuclear, so let's drop that column
# (leave out parts with structural zeros)
entity_tables['Oceania'] = entity_tables['Oceania'].drop('Nuclear', axis=1)

# The composition has 9 parts, so when we have 0 in one part it is more than 
# 10% already → parametric replacement
entity_tables_replaced = {}
for key in entity_tables:
    entity_tables_replaced[key] = entity_tables[key].coda.zero_replacement(n_samples=5000)
del(key)
    
# It seems like coda.zero_replacement closes the composition to 1

# %% Raw data visualization

# Define color set
colors_set = plt.cm.get_cmap('Set1')
colors_dict = {}
for i, col in enumerate(entity_tables['Africa']):
    colors_dict[col] = colors_set(i)
del(i, col)
colors_dict['Wind'] = '#2eb8b8' #Change because yellow can't be seen

# Let's plot each timeseries in usual R space
# Set raw data visualization location
plot_paths = {}
plot_paths['RawData'] = os.path.join(os.pardir, 'plots', 'raw')

# PLot
plot_entity_data(entity_tables, colors_dict, plot_paths['RawData'], y_label='Energy production per capita (kWh)')

# %% Percentage visualization

# Closure to 100%
entity_tables_perc = {}
for key in entity_tables_replaced:
    entity_tables_perc[key] = entity_tables_replaced[key].coda.closure(100)
del(key)
    
# Set percentage data visualization location
plot_paths['RawPercentage'] = os.path.join(os.pardir, 'plots', 'rawPercentage')

# Plot
plot_entity_data(entity_tables_perc, colors_dict, plot_paths['RawPercentage'], units='%')

# %% ILR default psi

entity_tables_ILR = {}
for key in entity_tables_perc:
    entity_tables_ILR[key] = entity_tables_perc[key].coda.ilr()
del(key)

for i, col in enumerate(entity_tables_ILR['Africa']):
    colors_dict[col] = colors_set(i)
del(i, col)

# Set ILR visualization location
plot_paths['ILR'] = os.path.join(os.pardir, 'plots', 'ILR')

# Plot
plot_entity_data(entity_tables_ILR, colors_dict, plot_paths['ILR'], units='ILR')

# %% ILR custom psi

# The binary partition tries to first split sustainable vs non, then removing 
# the energy sources that seem less related with the rest
partition_table = np.array([[1, 1, 1, 1, 1, 1, 1, 1, -1], 
                            [1, 1, 1, -1, -1, -1, -1, -1, 0], 
                            [1, -1, 1, 0, 0, 0, 0, 0, 0], 
                            [1, 0, -1, 0, 0, 0, 0, 0, 0], 
                            [0, 0, 0, 1, 1, 1, 1, -1, 0], 
                            [0, 0, 0, -1, 1, 1, 1, 0, 0], 
                            [0, 0, 0, 0, -1, 1, 1, 0, 0], 
                            [0, 0, 0, 0, 0, 1, -1, 0, 0]])

# Oceania needs a partition table which does not consider nuclear
partition_table_oceania = np.delete(partition_table, 3, axis=1)
partition_table_oceania = np.delete(partition_table_oceania, 5, axis=0)

# You can also use sbp_basis in extra.py ####################
psi = contrast_matrix(partition_table)
psi_oceania = contrast_matrix(partition_table_oceania)

# Split oceania from the rest because it has a different psi
entity_tables_perc_oceania = entity_tables_perc['Oceania']
entity_tables_perc.pop('Oceania')

# Calculate ILR
entity_tables_ILR = {}
for key in entity_tables_perc:
    entity_tables_ILR[key] = entity_tables_perc[key].coda.ilr(psi)
del(key)
entity_tables_ILR['Oceania'] = entity_tables_perc_oceania.coda.ilr(psi_oceania)

# Add back Oceania 
entity_tables_perc['Oceania'] = entity_tables_perc_oceania

# Set ILR visualization location
plot_paths['ILRCustom'] = os.path.join(os.pardir, 'plots', 'ILR_custom')

# Plot
plot_entity_data(entity_tables_ILR, colors_dict, plot_paths['ILRCustom'], units='ILR')

# %% Ordinary least square regression on ILR custom psi

prediction_range = range(1985, 2031)

linear_prediction = {}
for key in entity_tables_ILR:
    linear_prediction[key] = OLS(entity_tables_ILR[key], prediction_range)
del(key)

# %% ILR Regression visualization

plot_paths['ILRRegression'] = os.path.join(os.pardir, 'plots', 'ILR_regression')
plot_entity_data(entity_tables_ILR, colors_dict, plot_paths['ILRRegression'], pred=linear_prediction)

# %% Inverse ILR and plot regression

# Split Oceania because of different psi
linear_prediction_oceania = linear_prediction['Oceania']
linear_prediction.pop('Oceania')

# pycoda does not apply closure after inverse ILR, so I'll do it here
linear_prediction_R = {}
for key in linear_prediction:
    linear_prediction_R[key] = linear_prediction[key].coda.ilr_inv(psi)
    linear_prediction_R[key] = linear_prediction_R[key].coda.closure(100)
    linear_prediction_R[key].index = prediction_range
linear_prediction_R['Oceania'] = linear_prediction_oceania.coda.ilr_inv(psi_oceania)
linear_prediction_R['Oceania'] = linear_prediction_R['Oceania'].coda.closure(100)
linear_prediction_R['Oceania'].index = prediction_range

# Plots
plot_paths['inverseILR'] = os.path.join(os.pardir, 'plots', 'inverse_ILR')
plot_entity_data(entity_tables_perc, colors_dict, plot_paths['inverseILR'], units='%', pred=linear_prediction_R)

# %% Ternary plot

# Since we need three variables only, we need to amalgamate.
# NB: We begin again from raw data here, then we'll perform again zero-handling
# and closure

# I will not use Other, which we don't know what includes and 
# I wouldn't know where to amalgamate it

# Then, let's amalgamate classic polluting sources together (Coal, Gas, Oil), 
# then obviously green resources (Hydro, Wind, Solar) and in the end the ones 
# which are not so obvious (Nuclear and Bioenergy)

ternary_tables = {}
for key in entity_tables.keys() - {'Oceania'}:
    ternary_tables[key] = pd.DataFrame()
    ternary_tables[key]['Polluting'] = entity_tables[key]['Coal'] + entity_tables[key]['Gas']  + entity_tables[key]['Oil']
    ternary_tables[key]['Green'] = entity_tables[key]['Hydro'] + entity_tables[key]['Wind']  + entity_tables[key]['Solar']
    ternary_tables[key]['Alternative'] = entity_tables[key]['Nuclear'] + entity_tables[key]['Bioenergy']

ternary_tables['Oceania'] = pd.DataFrame()
ternary_tables['Oceania']['Polluting'] = entity_tables['Oceania']['Coal'] + entity_tables['Oceania']['Gas']  + entity_tables['Oceania']['Oil']
ternary_tables['Oceania']['Green'] = entity_tables['Oceania']['Hydro'] + entity_tables['Oceania']['Wind']  + entity_tables['Oceania']['Solar']
ternary_tables['Oceania']['Bioenergy'] = entity_tables['Oceania']['Bioenergy']

# The only country with some zeros after amalgamation is Oceania in Bioenergy
# So let's do some zero replacement on it
ternary_tables['Oceania'] = ternary_tables['Oceania'].coda.zero_replacement(n_samples=5000)

# Apply closure to 100
ternary_tables_perc = {}
for key in entity_tables_replaced:
    ternary_tables_perc[key] = ternary_tables[key].coda.closure(100)

# TODO: find a way to add labels
# Visualize each country in a ternary plot
# Each year is a sample
plot_paths['Ternary'] = os.path.join(os.pardir, 'plots', 'ternary')
plot_paths['TernaryCentered'] = os.path.join(os.pardir, 'plots', 'ternaryCentered')
for key in ternary_tables_perc:
    # Simple ternary plot
    tp = coda.plot.ternary(ternary_tables_perc[key], center=False)
    if not os.path.exists(plot_paths['Ternary']):
        os.makedirs(plot_paths['Ternary'])
    file = os.path.join(plot_paths['Ternary'], key)
    plt.savefig(file, dpi=300,  bbox_inches='tight')
    plt.close()
    
    # Centered ternary plot
    tp = coda.plot.ternary(ternary_tables_perc[key], center=True)
    if not os.path.exists(plot_paths['TernaryCentered']):
        os.makedirs(plot_paths['TernaryCentered'])
    file = os.path.join(plot_paths['TernaryCentered'], key)
    plt.savefig(file, dpi=300,  bbox_inches='tight')
    plt.close()
    
# %% PCA biplot

# The Biplot function centers, scales and applies CLR to data
# It is enough to feed it the closed data
plot_paths['PCAbiplot'] = os.path.join(os.pardir, 'plots', 'PCAbiplot')
for key in entity_tables_perc:
    mypca = coda.pca.Biplot(entity_tables_perc[key])
    mypca.plotscorelabels()
    # Save file
    if not os.path.exists(plot_paths['PCAbiplot']):
        os.makedirs(plot_paths['PCAbiplot'])
    file = os.path.join(plot_paths['PCAbiplot'], key)
    plt.savefig(file, dpi=300,  bbox_inches='tight')
    plt.close()
    
# The PCAs show the following correlations
# Classic: Coal, Gas, Oil, Hydro, Nuclear
# Modern: Solar, Wind
# Alternative: Bioenergy, Other
# So probably I should repeat the previous analysis using these partitions
# Modern and alternative are always simmetric to Classic, so for the ILR
# I should first split this way

# In every single graph the years until 1999 (included) and after 2000
# form two well distinguished clusters. I will split them in two ternary plot.
# In each plot, I will color the various countries with different colors,
# hhoping to see nice clusters and draw conclusions on energy usage
