"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from data import make_data1, make_data2
from plot import plot_boundary


# 1 Decision tree

if __name__ == "__main__":
	# Parameters
	make_data = [make_data1, make_data2]
	n_samples, train_size, n_gen = 2000, 150, 5
	depth = [1, 2, 4, 8, None]

	# 1.1 Decision boundary
	for i in range(len(make_data)):
		# Data set
		X, y = make_data[i](n_samples, random_state = 0)
		X_train, X_test, y_train, y_test = train_test_split(
			X, y,
			train_size = train_size,
			shuffle = False
		)

		for j in range(len(depth)):
			# Classifier
			dtc = DecisionTreeClassifier(max_depth = depth[j])
			dtc.fit(X_train, y_train)

			# Plot
			plot_boundary(
				"make_data" + str(i + 1) + "_depth" + str(depth[j]),
				dtc,
				X_test[0:train_size],
				y_test[0:train_size],
				title = "\\texttt{max\_depth = " + str(depth[j]) + "}"
			)

	# 1.2 Accuracies
	print("make_data", "max_depth", "mean", "std")

	for i in range(len(make_data)):
		for j in range(len(depth)):
			accr = np.empty(n_gen)

			for k in range(n_gen):
				# Data set
				X, y = make_data[i](n_samples, random_state = k)
				X_train, X_test, y_train, y_test = train_test_split(
					X, y,
					train_size = train_size,
					shuffle = False
				)

				# Classifier
				dtc = DecisionTreeClassifier(max_depth = depth[j])
				dtc.fit(X_train, y_train)

				# Accuracy
				accr[k] = dtc.score(X_test, y_test)

			print(i + 1, depth[j], "%f" % np.mean(accr), "%f" % np.std(accr))
