# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 09:43:30 2016

@author: anmason

This code illustrates a common use of logistic regression, which is to predict the choice of consumers. In this simple example,
we have the case of a product which is being sold at variaous points of price and performace, and consumers have two choices:
    1. Buy (coded as '1')
    2. Don't buy (coded as '0')

In the data file provided, we have the following fields:
    1. Price
    2. Performance
    3. Choice: This is the 'actual' choice that consumers made, which includes some randomness
    4. Choice_Test: This is the 'perfect' choice if no randomness existed in the model. (derived from a perfect model formula)
        The parameters of the 'perfect model' are: beta_price=-0.65; beta_performance=2.0

The objective of this code is to explore how close we can get to the actual model used to simulate the data, using
logistic regression.

* The data used in this example was simulated using simple spreadsheet formulas, also provided in this folder in Logistic_Data.xlsx
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as log_reg


def main():
    dir_loc = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_loc, 'Data3')

    in_data_path = os.path.join(data_path, 'Logistic_Data.csv')
    data_table = pd.read_csv(in_data_path)
    X = data_table[['Price', 'Performance']].values
    y_rnd = data_table[['Choice']].values
    y_test = data_table['Choice_Test'].values

    # See http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    logit = log_reg.Logit(y_rnd, X)
    fitted_model = logit.fit()
    parameters = fitted_model.params
    y_pred_val = fitted_model.predict(X)
    y_pred = np.where(y_pred_val > 0.5, 1, 0)

    error = abs(y_test - y_pred)

    data_table['predict_value'] = y_pred_val
    data_table['prediction'] = y_pred
    data_table['error'] = error

    out_data_path = os.path.join(data_path, 'Output_logistic.csv')
    data_table.to_csv(out_data_path)


if __name__ == '__main__':
    main()
