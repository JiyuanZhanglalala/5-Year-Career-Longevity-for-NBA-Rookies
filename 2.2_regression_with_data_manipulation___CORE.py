# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 09:43:30 2016

@author: anmason

Sometimes when fitting a model, using two variables is not sufficient; there may be relationships to other variables that
could make a much better prediction model if included.

For example, in the case of the basketball players, it is possible that the position of the player may influence their weight;
this code attempts to include some additional, transformed variables.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


def pandas_operations(data_sub):
    data_sub['H_sq'] = data_sub['Height']**2
    data_sub[['C', 'F', 'T']] = pd.get_dummies(data_sub['Pos'])  # note the array! [[]] you are geting list output
    data_sub['Constant'] = 1

    return data_sub


def main():
    dir_loc = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_loc, 'Data')
    data_path = os.path.join(data_path, 'Sample_data.csv')
    data = pd.read_csv(data_path, encoding = "ISO-8859-1")

    # Data Fields: Player	Pos	Height	Weight	Age
    data_refined = pandas_operations(data)

    # Note our data now includes also: 'H_sq','C','F','T'
    # The following array "x_labels" controls what variables ypu are including into the model. Use this to make manual edits to your models
    x_labels = ['Constant', 'Height', 'H_sq', 'C', 'F', 'Age']  # Note we remove 'T'
    x_mat = data_refined[x_labels].values
    y_obs = data_refined['Weight'].values

    t_model = sm.OLS(y_obs, x_mat)
    results = t_model.fit()
    parameters = results.params
    p_values = results.pvalues
    r_sq = results.rsquared
    r_sq_adj = results.rsquared_adj

    print((p_values * 100))
    print(results.summary())


if __name__ == '__main__':
    main()
