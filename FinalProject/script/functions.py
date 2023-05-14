# -*- coding: utf-8 -*-
"""
Created on Mon May  8 19:05:10 2023

@author: pf259
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import statsmodels.api as sm
import ternary

def get_entity_dict(df):
    '''
    Split dataset in different countries
    '''
    entity_dict = {}
    for entity in df['Entity'].unique():
        entity_dict[entity] = df[df['Entity']== entity].iloc[:, 1:]
        entity_dict[entity].set_index('Year', inplace=True)
        entity_dict[entity].columns = [col.split()[0] for col in entity_dict[entity].columns]
    return entity_dict

def plot_entity_data(entity_dict, colors_dict, destination_folder=None, y_label='', pred=None):
    '''
    Plot timeseries
    '''
    for key in entity_dict:
        df = pd.DataFrame(entity_dict[key])
        colors = [colors_dict[col] for col in df.columns]
        ax = df.plot(title=key, figsize=(10, 5), color=colors)
        # Plot predictions
        if pred is not None:
            pred[key].plot(ax=ax, color=colors, linestyle='--', legend=None)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), labels=df.columns)
        plt.suptitle('Energy resources')
        plt.ylabel(y_label)
        # Save plot
        if destination_folder is not None:
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
            file = os.path.join(destination_folder, key)
            plt.savefig(file, dpi=300,  bbox_inches='tight')
        plt.close()
    del(key)
    
def contrast_matrix(binary_partition_table):
    '''
    Define contrast matrix from partition table
    '''
    table = copy.deepcopy(binary_partition_table)
    table = table.astype(float)
    r = np.sum(table == 1, axis=1)
    s = np.sum(table == -1, axis=1)
    a_plus = 1/r * np.sqrt(r * s / (r + s))
    a_minus = -1/s * np.sqrt(r * s / (r + s))
    for i in range(table.shape[0]):
        table[i] = np.where(table[i] == 1, a_plus[i], table[i])
        table[i] = np.where(table[i] == -1, a_minus[i], table[i])
    del(i)
    return table

def OLS(df, prediction_range):
    '''
    Oridnary least squares fit and prediction in the specified range
    '''
    X = df.index
    X = sm.add_constant(X)
    X_pred = sm.add_constant(prediction_range)
    df_pred = pd.DataFrame(index=X_pred[:,1].astype(int))
    for col in df.columns:
        y = df[col]
        model = sm.OLS(y, X).fit()
        
        # Make predictions using the OLS model
        y_pred = model.predict(X_pred)
        df_pred[col] = y_pred
    del(col)
    return df_pred

def plot_ternary(dic, colors,  center=False, destination_folder=None, name='ternary'):
    '''
    Ternary plot
    dic: dictionary with a df in each key. The key is the category
    colors: dictionary with colors associated to each key of dic
    '''
    fig, tax = ternary.figure(scale=100)
    labels = dic[list(dic.keys())[0]].columns

    # Draw Boundary and Gridlines
    tax.boundary(linewidth=1.5)
    
    # Set Axis labels and Title
    tax.left_axis_label("% "+ labels[2],
                        fontsize=16,
                        offset=0.14)
    tax.right_axis_label("% "+ labels[1],
                         fontsize=16,
                         offset=0.14)
    tax.bottom_axis_label("% "+ labels[0],
                          fontsize=16,
                          offset=0.12)
    
    # Plot grid and ticks only for not centered data
    if not center:    
        tax.gridlines(multiple=10, color="black", linestyle= '--')
        tax.ticks(axis='lbr',
                  linewidth=1,
                  multiple=10,
                  offset=0.03)
        
    # If centered, calculate the geometric mean for all data
    if center:
        tax.ticks(axis='lbr',
                  linewidth=1,
                  multiple=100,
                  offset=0.03)
        data = pd.DataFrame()
        for key in dic:
            data = pd.concat([data, dic[key]])
        gmean = data.coda.gmean()
        
    # Remove default Matplotlib Axes
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
        
    # Plot data
    for key in dic:
        # Centering
        if center:
            sdata = (dic[key]/gmean).coda.closure(100)
        else:
            sdata = dic[key]
        points = [tuple(x) for x in sdata.to_records(index=False)]
        alpha = np.linspace(0.1, 1, len(dic[key])) #Sequential transparency
        for i in range(len(dic[key]) - 1):
            tax.scatter([points[i]],
                        color=colors[key],
                        alpha=alpha[i])
        # Plot only last point with legend
        tax.scatter([points[-1]],
                    color=colors[key],
                    alpha=alpha[-1],
                    label=key)
    
    tax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
    plt.suptitle(name)
    
    # Save plot
    if destination_folder is not None:
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        file = os.path.join(destination_folder, name)
        plt.savefig(file, dpi=300,  bbox_inches='tight')
    plt.close()
