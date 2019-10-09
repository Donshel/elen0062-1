"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

from data import make_data1, make_data2
from plot import plot_boundary


# 3 Naive Bayes classifier

# 3.2 Classifier
class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):

    def _more_tags(self):
        return {'requires_fit': True}

    def fit(self, X, y):
        """Fit a Gaussian navie Bayes model using the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Classes
        self.classes_, y = np.unique(y, return_inverse = True)
        self.n_classes = len(self.classes_)

        # Variables
        self.n_variables = X.shape[1]

        # Prior
        self.prior = np.bincount(y)

        # Means and variances
        self.mu = np.empty((self.n_classes, self.n_variables))
        self.var = np.copy(self.mu)

        for i in range(self.n_classes):
            self.mu[i][:] = np.mean(X[y == i, :], axis = 0)
            self.var[i][:] = np.var(X[y == i, :], axis = 0)

        # Return the classifier
        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        return self.classes_[np.argmax(self.predict_proba(X, False), axis = 1)]

    def predict_proba(self, X, normalize = True):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        # Check is fit had been called
        check_is_fitted(self, ["classes_"])

        # Check that X has correct shape
        check_array(X)
        if X.shape[1] != self.n_variables:
            raise ValueError("X must have %d variables" % self.p)

        # Probabilities
        y = np.empty((X.shape[0], self.n_classes))

        for n in range(X.shape[0]):
            for i in range(self.n_classes):
                temp = self.mu[i] - X[n]
                temp = temp ** 2
                temp = temp / self.var[i]
                temp = np.exp(-np.sum(temp) / 2)
                temp = temp / np.sqrt(np.prod(self.var[i]))
                temp = self.prior[i] * temp
                
                y[n][i] = temp

            if normalize:
                y[n] /= np.sum(y[n])

        return y

if __name__ == "__main__":
    # Parameters
    make_data = [make_data1, make_data2]
    m, n, g = 2000, 150, 5

    # 3.3 Accuracies
    print("make_data", "mean", "std")

    for i in range(len(make_data)):
        # Generations
        X, y = [None] * g, [None] * g
        for j in range(g):
            X[j], y[j] = make_data[i](m, random_state = j)

        # Classifiers
        A = np.zeros(g)

        for j in range(g):
            nbc = NaiveBayesClassifier()
            nbc.fit(X[j][0:n], y[j][0:n])

            A[j] = nbc.score(X[j][n:m], y[j][n:m])
        
        print(str(i + 1), "%f" % np.mean(A), "%f" % np.std(A))

    # 3.4 Plots
    for i in range(len(make_data)):
        # Generation
        X, y = make_data[i](m, random_state = 0)

        # Classifier
        nbc = NaiveBayesClassifier()
        nbc.fit(X[0:n], y[0:n])

        st = "make_data" + str(i + 1) + "_naivebayes"
        plot_boundary(st, nbc, X[0:n], y[0:n])
