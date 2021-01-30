# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 09:43:30 2016

@author: anmason


In this module we make some basic data manipulations and create a regression of height v.s weight of basketball players.
This module runs linearly and with the purpose of illustrating:
    - Simple data manipulation, putting data into simple numeric arrays
    - Calling a statistical function to perform the corresponding calculations
    - Extracting data from the model created
    - Making predictions on new data
    - Saving results as a figure/csv file
This module is the foundation to manipulating data in python and understanding upcoming analysis.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


def main():
    # The following are used to declare the location of the code-file and data;
    # then read from the location declared.
    dir_loc = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_loc, 'Data')
    data_read_path = os.path.join(data_path, 'Sample_data.csv')
    data = pd.read_csv(data_read_path, encoding = "ISO-8859-1")

    # Data Fields: Player	Pos	Height	Weight	Age
    # Data is contained in a pandas object
    data_x = data['Height']
    data_y = data['Weight']

    # Here data is placed in a numpy object from pandas
    x = data_x.values
    yd = data_y.values

    # Operation necesary to do a regression using statsmodels
    # Also see:  http://statsmodels.sourceforge.net/devel/regression.html
    xd = sm.add_constant(x)
    t_model = sm.OLS(yd, xd)
    results = t_model.fit()
    interc, slope = results.params

    # The prediction data can be obtained from the following statsmodels methods
    # Remember that "results" is a statsmodels object and will give back desired statistics
    y_pred = results.predict()
    residuals = results.resid
    results.summary()
    r_sq = results.rsquared
    results.rsquared_adj

    # Now lets predict some additional data; first read the new dataset, then edit and input the predictors
    pred_data_read_path = os.path.join(data_path, 'Predict_data.csv')
    pred_data = pd.read_csv(pred_data_read_path)

    # Same functions as above
    pred_x = pred_data['Height']
    x2 = pred_x.values
    x_pr = sm.add_constant(x2)
    y_pred_2 = results.predict(x_pr)

    # To finalize, lets make a graph for the regression results:
    plt.figure()
    plt.scatter(x, yd, s=30, alpha=0.15, marker='.')
    plt.plot(x2, y_pred_2, '-r', marker='x')
    plt.title('Height vs Weight Plot')
    plt.tick_params(axis='both', which='major')
    plt.ylabel('Weight')
    plt.xlabel('Height')

    fig_text = 'intercept = ' + str(interc) + '\nslope = ' + str(slope) + '\nr-sq = ' + str(r_sq)
    plt.annotate(fig_text, xy=(0.95, 0.05), xycoords='axes fraction', fontsize=10, horizontalalignment='right', verticalalignment='bottom')

    # save figure to working directory
    path = os.path.join(data_path, '1.0_simple_regression_plots.png')
    plt.savefig(path)
    plt.close()

    # save prediction data to working directory
    pred_data['Predicted Weight'] = y_pred_2
    path = os.path.join(data_path, '1.0.Predict_data_results.csv')
    pred_data.to_csv(path)


if __name__ == '__main__':
    main()
