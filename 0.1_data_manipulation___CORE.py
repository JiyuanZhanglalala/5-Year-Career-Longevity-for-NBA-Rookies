"""
Created on Sat Nov 12 09:43:30 2016

@author: anmason

The purpose of this sample code is to illustrate some of the basic pandas python functionality for:
    - Data read/write using pandas
    - Data manipulations using numpy
    - data manipulations using pandas
This module is the foundation to manipulating data in python and understanding upcoming analysis.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


def numpy_tests(x_sub):
    x_square = x_sub**2
    x_min = np.min(x_sub)
    x_norm = np.sqrt(np.dot(x_sub, x_sub))
    x_norm = np.linalg.norm(x_sub)
    x_class = np.where(x_sub < 75, 1, 0)

    return x_square, x_class, x_min, x_norm


def pandas_tests(data_sub):
    data_sub['H_sq'] = data_sub['Height']
    data_sub['H_min'] = data_sub['Height'].min()
    data_sub['Norm'] = np.linalg.norm(data_sub['Height'])
    data_sub['H_Class'] = np.where(data_sub['Height'] < 75, 1, 0)

    return data_sub


def main():
    dir_loc = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_loc, 'Data')
    path = os.path.join(data_path, 'Sample_data.csv')
    data = pd.read_csv(path, encoding = "ISO-8859-1")

    # Data Fields: Player	Pos	Height	Weight	Age
    data_x = data['Height']
    data_y = data['Weight']

    # Here data is placed in a numpy object from pandas
    x = data_x.values

    x_square, x_class, x_min, x_norm = numpy_tests(x)
    data_refined = pandas_tests(data)

    path = os.path.join(data_path, '0.1.Sample_data_modified.csv')
    data_refined.to_csv(path)
    print(data_refined)

if __name__ == "__main__":
    main()
