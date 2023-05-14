# -*- coding: utf-8 -*-
"""
Created on Tue May  2 18:20:57 2023

@author: pf259
"""

import os
import pandas as pd
import numpy as np
import pycodamath as coda

#%% Load raw data
# Get the directory path of the current script file
dir_path = os.path.dirname(os.path.realpath(__file__))

# Set the working directory to the script directory
os.chdir(dir_path)

# Set raw data location
raw_data_location = os.path.join(os.pardir, 'data', 'World_Energy_sources.csv')

# Read raw data
raw_data = pd.read_csv(raw_data_location, delimiter=';')


#%% Count zeros
# Group dataset by country column
grouped = raw_data.groupby('Entity')

# Define function to count zeros, NaNs, and zero + NaN combinations
def count_zeros_nans(x):
    zeros = np.count_nonzero(x == 0)
    nans = np.isnan(x).sum()
    zero_nans = zeros + nans
    return pd.Series([zeros, nans, zero_nans], index=["Zeros", "NaNs", "Zeros+NaNs"])

# Apply function to each group and aggregate results
counts = grouped.apply(lambda x: x.iloc[:, 2:].apply(count_zeros_nans)).stack()

# Define function to count zeros, NaNs, and zero + NaN combinations as percentages
def count_zeros_nans_pct(x):
    total = len(x)
    zeros = np.count_nonzero(x == 0) / total * 100
    nans = np.isnan(x).sum() / total * 100
    zero_nans = zeros + nans
    return pd.Series([zeros, nans, zero_nans], index=["Zeros%", "NaNs%", "Zeros+NaNs%"])

# Apply function to each group and aggregate results
counts_pct = grouped.apply(lambda x: x.iloc[:, 2:].apply(count_zeros_nans_pct)).stack()

#%% Time series side by side for each variable, and lowest positive values

# Define a custom function to calculate the lowest positive value
def lowest_positive(series):
    # Filter out negative values
    positive_series = series[series > 0]
    if len(positive_series) == 0:
        # If all values are negative, return NaN
        return 0
    else:
        # Otherwise, return the lowest positive value
        return positive_series.min()
    
# Get the list of variable names
variables = list(raw_data.columns[2:])

# Create a dictionary to store the aligned tables
aligned_tables = {}

# Iterate over the variables and pivot the dataset for each variable
for var in variables:
    # Pivot the dataset
    aligned_raw_data = raw_data.pivot(index="Year", columns="Entity", values=var)
    
    # Calculate the lowest positive value for each country
    lowest_positive_values = aligned_raw_data.apply(lowest_positive)
    
    # Store the aligned table and lowest values in the dictionary
    aligned_tables[var] = {"aligned_table": aligned_raw_data, "lowest_positive_values": lowest_positive_values}
    
# Get the list of country names
countries = list(raw_data["Entity"].unique())

# Create an empty DataFrame to store the lowest positive values
lowest_positive_table = pd.DataFrame(index=countries)

# Iterate over the variables and add the corresponding lowest positive values to the table
for var in aligned_tables.keys():
    lowest_positive_values = aligned_tables[var]["lowest_positive_values"]
    lowest_positive_table[var] = lowest_positive_values
    del(lowest_positive_values)
    
# %% Zero replacement
raw_data.fillna(0, inplace=True)  # Substitute nans with zeros 
replaced_zeros = raw_data.iloc[:, 2:].coda.zero_replacement(n_samples=5000)

