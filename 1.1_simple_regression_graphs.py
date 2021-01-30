# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 09:43:30 2016

@author: anmason


In this optional module, we expand on the previous regression analysis to make graphs that are more involved
    - Plot histograms of both regression variables
    - Plot residuals (errors) of the regression

"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


def graph_regression_data(x1, yd, y_pred, directory="", figure_name='fit.png'):
    # Set some commonly used variables in the top
    num_bins = 12

    # Set some of the input/output data here
    resid = yd - y_pred
    capab_pr = np.average(yd) + resid

    #  Note: we create fig and under fig, we create sublots. Here we declare them
    #  Note: When dealing with subplots, you can still change the title, axes, etc.
    # Use the set_XXX method to use figure commands in subplots.
    # i.e. for a figure we use fig.title(); for subplots we use sub.set_title ()
    fig = plt.figure()
    sub1 = fig.add_subplot(221)
    sub2 = fig.add_subplot(222)
    sub3 = fig.add_subplot(223)
    sub4 = fig.add_subplot(224)

    # Create subplot 3
    sub3.scatter(x1, yd, s=30, alpha=0.15, marker='.')
    sub3.plot(x1, y_pred, '-r')
    sub3.set_title('Height vs Weight Plot')
    sub3.set_ylabel('Weight')
    sub3.set_xlabel('Height')

    # Create subplot 2
    sub2.hist(capab_pr, bins=num_bins)
    sub2.set_title('Residual Histogram')

    # Create subplot 1
    sub1.hist(x1, bins=num_bins)
    sub1.set_title('Height Histogram')

    # Create subplot 4
    sub4.hist(yd, bins=num_bins, orientation="horizontal")
    sub4.set_title('Weight histogram')

    fig.tight_layout()    # prevents text in figures from overlapping
    fig.savefig(os.path.join(directory, figure_name))
    plt.close()


def main():
    # The following are used to declare the location of the code-file and data;
    # then read from the location declared.
    dir_loc = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_loc, 'Data')
    data_read_path = os.path.join(data_path, 'Sample_data.csv')
    data = pd.read_csv(data_read_path, encoding = "ISO-8859-1")

    # Data Fields: Player	Pos	Height	Weight	Age
    data_x = data['Height']
    data_y = data['Weight']

    # Here data is placed in a numpy object from pandas
    x = data_x.values
    yd = data_y.values

    # Operation necesary to do a regression using statsmodels
    xd = sm.add_constant(x)
    t_model = sm.OLS(yd, xd)
    results = t_model.fit()
    interc, slope = results.params

    # This is used to obtain the predicted values and errors from the regression
    y_pred = results.predict()

    graph_regression_data(x, yd, y_pred, data_path, 'fitting_graphs.png')


if __name__ == '__main__':
    main()
