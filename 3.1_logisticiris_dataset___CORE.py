# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 09:43:30 2016

@author: anmason

In this example, we use a logistic regression classifier from sklearn to build a model for predicting the type of flower in a
standard sample dataset.

*NOTE: The following code is take from http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html.
 and was modified for the purposes of teaching this class
    For more information see: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression

**NOTE: In order to run this code, you will need to install a new python package using pip. To do this, do the following:
    1. Open a command prompt window by:
        WINDOWS: press the come button and type cmd
        MAC OS: use Spotlight to search for "Terminal"
    2. In the command window, write the following:
        pip install sklearn
    3. sklearn should proceed to install automatically in your machine
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets


def main():

    # Import data from the provided dataframe
    dir_loc = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_loc, 'Data_Iris')
    iris_df = pd.read_csv(os.path.join(data_path, 'iris_dataset.csv'))
    # feature_selection = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']  # these are all the features
    feature_selection = ['sepal length (cm)', 'sepal width (cm)']  # this way we try only the first two
    X = iris_df[feature_selection].values  # we filter the features here
    Y = iris_df['target'].values

    logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    # we create an instance of Neighbours Classifier and fit the data.
    logreg.fit(X, Y)
    y_predicted = logreg.predict(X)
    iris_df['target_predicted'] = y_predicted
    correct_predictions = np.where(y_predicted == Y, 1, 0)
    accuracy = np.average(correct_predictions)
    print('prediction accuracy: {}'.format(accuracy))
    iris_df.to_csv(os.path.join(data_path, '3.1.iris_dataset_predictions.csv'))

    if len(X[0]) == 2:
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = .02  # step size in the mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1, figsize=(4, 3))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlabel(feature_selection[0])
        plt.ylabel(feature_selection[1])
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.show()
        plt.savefig(os.path.join(data_path, 'data_split.png'))


if __name__ == '__main__':
    main()
