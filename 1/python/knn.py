"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from data import make_data1, make_data2
from plot import plot_boundary, plot_xy


# 2 K-nearest neighbors

if __name__ == "__main__":
	# Parameters
	make_data = [make_data1, make_data2]
	n_samples, train_size, n_fold = 2000, 150, 10
	neighbors = [1, 5, 10, 75, 100, 150]

	# 2.1 Decision boundary
	for i in range(len(make_data)):
		# Data set
		X, y = make_data[i](n_samples, random_state = 0)
		X_train, X_test, y_train, y_test = train_test_split(
			X, y,
			train_size = train_size,
			shuffle = False
		)

		for j in range(len(neighbors)):
			# Classifier
			knc = KNeighborsClassifier(n_neighbors = neighbors[j])
			knc.fit(X_train, y_train)

			# Plot
			plot_boundary(
				"make_data" + str(i + 1) + "_neighbors" + str(neighbors[j]),
				knc,
				X_test[0:train_size],
				y_test[0:train_size],
				title = "\\texttt{n\_neighbors = " + str(neighbors[j]) + "}"
			)


	# 2.2 K-fold cross validation
	neighbors = np.array(range(1, 101))
	accr = np.empty((len(neighbors), n_fold))

	# Data set
	X, y = make_data[1](n_samples, random_state = 0)
	kf = KFold(n_splits = n_fold)

	for i in range(len(neighbors)):
		for j, (train_index, test_index) in enumerate(kf.split(X)):
			# Fold
			X_train, y_train = X[train_index], y[train_index]
			X_test, y_test = X[test_index], y[test_index]

			# Classifier
			knc = KNeighborsClassifier(n_neighbors = neighbors[i])
			knc.fit(X_train, y_train)

			# Accuracy
			accr[i, j] = knc.score(X_test, y_test)

	# Maximum
	accr = accr.mean(axis = 1)
	i = np.argmax(accr)
	print("The maximum accuracy %f is reached for n_neighbors = %d." % (accr[i], neighbors[i]))

	# Plot
	plot_xy(
		"make_data2_kfold",
		neighbors,
		accr,
		"\\texttt{n\_neighbors}",
		"Accuracy"
	)
