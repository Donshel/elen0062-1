"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

from data import make_data1, make_data2
from plot import plot_boundary


# 2 K-nearest neighbors

if __name__ == "__main__":
	# Parameters
	make_data = [make_data1, make_data2]
	m, n, f = 2000, 150, 10
	K = [1, 5, 10, 75, 100, 150]

	# 2.1 Decision boundary
	for i in range(len(make_data)):
		# Generation
		X, y = make_data[i](m, random_state = 0)

		# Classifiers
		for k in K:
			knc = KNeighborsClassifier(n_neighbors = k)
			knc.fit(X[0:n], y[0:n])

			st = "make_data" + str(i + 1) + "_neighbors" + str(k)
			plot_boundary(st, knc, X[0:n], y[0:n])

	# 2.2 K-fold cross validation
	print("make_data", "n_neighbors", "mean", "std")

	for i in range(len(make_data)):
		# Generation
		X, y = make_data[i](m, random_state = 0)
		kf = KFold(n_splits = f)

		# Classifiers
		for k in range(5, K[-1], 5):
			A = np.zeros(f)

			for j, (train, test) in enumerate(kf.split(X)):
				knc = KNeighborsClassifier(n_neighbors = k)
				knc.fit(X[train], y[train])

				A[j] = knc.score(X[test], y[test])

			print(str(i + 1), str(k), "%f" % np.mean(A), "%f" % np.std(A))
