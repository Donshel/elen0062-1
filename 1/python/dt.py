"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.tree import DecisionTreeClassifier

from data import make_data1, make_data2
from plot import plot_boundary


# 1 Decision tree

if __name__ == "__main__":
	# Parameters
	make_data = [make_data1, make_data2]
	m, n, g = 2000, 150, 5
	D = [1, 2, 4, 8, None]

	# 1.1 Decision boundary
	for i in range(len(make_data)):
		# Generation
		X, y = make_data[i](m, random_state = 0)

		# Classifiers
		for d in D:
			dtc = DecisionTreeClassifier(max_depth = d)
			dtc.fit(X[0:n], y[0:n])

			st = "make_data" + str(i + 1) + "_depth" + str(d)
			#plot_boundary(st, dtc, X[0:n], y[0:n])

	# 1.2 Accuracies
	print("make_data", "max_depth", "mean", "std")

	for i in range(len(make_data)):
		# Generations
		X, y = [None] * g, [None] * g
		for j in range(g):
			X[j], y[j] = make_data[i](m, random_state = j)

		# Classifiers
		for d in D:
			A = np.zeros(g)

			for j in range(g):
				dtc = DecisionTreeClassifier(max_depth = d)
				dtc.fit(X[j][0:n], y[j][0:n])

				A[j] = dtc.score(X[j][n:m], y[j][n:m])

			print(str(i + 1), str(d), "%f" % np.mean(A), "%f" % np.std(A))
