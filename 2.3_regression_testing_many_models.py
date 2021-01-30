# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 09:43:30 2016

@author: anmason

Very often when using computers for mathematical computation, it is to automate a process that must be repeated
many times. For instance, in the case of basketball players, we have five possible variables that can explain weight; if
we were to explore all possible combinations of these variables, we would have to manually try out 2^5 = 32 models!. This may not
be feasible in practice.

This module shows a simple way to try out many drifferent models and export the key descriptive statistics into a csv file.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


def pandas_tests(data_sub):
    data_sub['H_sq'] = data_sub['Height']**2
    data_sub[['C', 'F', 'T']] = pd.get_dummies(data_sub['Pos'])  # note the array! [[]] you are geting list output
    data_sub['Constant'] = 1

    return data_sub


def main():
    dir_loc = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_loc, 'Data')

    sample_data_path = os.path.join(data_path, 'Sample_data.csv')
    data = pd.read_csv(sample_data_path, encoding = "ISO-8859-1")

    mod_data_path = os.path.join(data_path, 'Testing_models.csv')
    mods_to_test = pd.read_csv(mod_data_path, encoding = "ISO-8859-1")

    # Data Fields:   Player	     Pos	    Height	Weight	Age
    data_refined = pandas_tests(data)

    x_labels = ['Constant', 'Height', 'C', 'F', 'Age']  # Note we remove 'T'
    x_labels_p = ['Constant_pval', 'Height_pval', 'C_pval', 'F_pval', 'Age_pval']
    x_mat = data_refined[x_labels].values
    y_obs = data_refined['Weight'].values

    test_vectors = mods_to_test.values
    data_container = [0] * (len(test_vectors))
    data_labels = x_labels_p + x_labels + ['R-sq-adj']
    for idx, subset in enumerate(test_vectors):
        x_mat_sub = x_mat * subset   # makes the columns where "subset=0" zero.  Using numpy, always check your columns match
        t_model = sm.OLS(y_obs, x_mat_sub)
        results = t_model.fit()
        parameters = np.round(results.params, 8)
        p_values = np.round(results.pvalues, 4)
        r_sq_adj = results.rsquared_adj
        data_container[idx] = p_values.tolist() + parameters.tolist() + [r_sq_adj]

    output_data = pd.DataFrame(data_container, columns=data_labels)
    out_data_path = os.path.join(data_path, '2.3.Output_for_multiple_models.csv')
    output_data.to_csv(out_data_path)


if __name__ == '__main__':
    main()
