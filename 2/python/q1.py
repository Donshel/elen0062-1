"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 2 - Bias and variance analysis
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.integrate as integrate


# 1 Bayes model and residual error in classification

if __name__ == "__main__":
	# Parameters

	sigma = np.sqrt(0.1)
	r_minus, r_plus = 1, 2

	r_pm = (r_minus + r_plus) / 2

	# Residual error

	f = lambda x, mu, sigma: np.exp(-(x - mu) ** 2 / 2 / sigma ** 2) / np.sqrt(2 * np.pi) / sigma
	g = lambda x: f(x, r_plus, sigma) - f(x, r_minus, sigma)

	E = 1 / 2 + 1 / 2 * integrate.quad(g, -r_pm, r_pm)[0]

	print("The residual error is %f" % E)
