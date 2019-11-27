"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 2 - Bias and variance analysis
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import choice, randn

from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

import os
from matplotlib import rc
from matplotlib import pyplot as plt

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 14})

# 3 Bias and variance estimation

if __name__ == "__main__":
	# X
	x_min, x_max = -10, 10
	x_dom = np.linspace(x_min, x_max, 200)

	# y
	sigma = 1 / 10
	f = lambda x: np.sin(x) * np.exp(-x ** 2 / 16)

	# make_data
	def make_data(n, q):
		X = x_dom[choice(len(x_dom), (n, q + 1))]
		y = f(X[:, 0]).reshape(n, 1) + sigma * randn(n, 1)

		return X, y

	# 3(d) Estimations

	def bias_variance(X, y, model, p=10):
		n = X.shape[0]
		dom, inv = np.unique(X, return_inverse=True, axis=0)

		E_y, V_y = np.empty(dom.shape[0]), np.empty(dom.shape[0])

		for i in range(len(dom)):
			# 1
			temp = y[inv == i]
			# 2
			E_y[i] = temp.mean()
			if temp.shape[0] > 1:
				V_y[i] = temp.var(ddof=1)
			else:
				V_y[i] = np.nan

		y_hat = np.empty((len(dom), p))

		# 3
		index = np.arange(n // p)
		for i in range(p):
			model.fit(X[index, :], y[index])
			y_hat[:, i] = model.predict(dom).reshape(len(dom))

			index += n // p

		# 4
		E_LS = y_hat.mean(axis=1)
		V_LS = y_hat.var(axis=1, ddof=1)

		# 5
		noise = V_y
		bias2 = (E_y - E_LS) ** 2
		variance = V_LS

		# 6
		noise_bis = noise.copy()
		noise_bis[np.isnan(noise)] = 0
		error = noise_bis + bias2 + variance

		return dom, noise, bias2, variance, error

	## Learning Set

	n = 10 ** 4
	q = 0

	X, y = make_data(n, q)

	### mkdir -p products/pdf
	dirs = "products/pdf/"
	os.makedirs(dirs, exist_ok=True)

	### f(x)

	plt.figure()
	plt.xlabel("$x$")
	plt.ylabel("$y$")
	plt.grid(True)
	plt.scatter(X[:100], y[:100], color='#ff7f0e')
	plt.plot(x_dom, f(x_dom))
	plt.savefig(dirs + "{}.pdf".format("data"), bbox_inches='tight')
	plt.close()

	## Ridge regression

	dom, noise, bias2, variance, error = bias_variance(X, y, Ridge(alpha=1.0))

	plt.figure()
	plt.xlabel("$x_r$")
	plt.grid(True)
	plt.semilogy(dom, noise, label = "noise$(x_r)$")
	plt.semilogy(dom, bias2, label = "bias$^2(x_r)$")
	plt.semilogy(dom, variance, label = "variance$(x_r)$")
	plt.semilogy(dom, error, label = "error$(x_r)$")
	plt.legend()
	plt.savefig(dirs + "{}.pdf".format("rrg"), bbox_inches='tight')
	plt.close()

	## K-Neighbors regression

	dom, noise, bias2, variance, error = bias_variance(X, y, KNeighborsRegressor(n_neighbors=5))

	plt.figure()
	plt.xlabel("$x_r$")
	plt.grid(True)
	plt.semilogy(dom, noise, label = "noise$(x_r)$")
	plt.semilogy(dom, bias2, label = "bias$^2(x_r)$")
	plt.semilogy(dom, variance, label = "variance$(x_r)$")
	plt.semilogy(dom, error, label = "error$(x_r)$")
	plt.legend()
	plt.savefig(dirs + "{}.pdf".format("knr"), bbox_inches='tight')
	plt.close()

	# 3(e) Mean estimations

	def mean_bias_variance(X, y, model):
		noise, bias2, variance, error = bias_variance(X, y, model)[1:]

		noise = noise[np.logical_not(np.isnan(noise))]

		return noise.mean() if noise.shape[0] > 0 else 0, bias2.mean(), variance.mean(), error.mean()

	## Ridge regression

	### n

	n = (10 ** np.arange(2.4, 5.2, 0.2)).astype(int)
	q = 0
	alpha = 0.001

	noise = np.empty(len(n))
	bias2 = np.empty(noise.shape)
	variance = np.empty(noise.shape)
	error = np.empty(noise.shape)

	X, y = make_data(n[-1], q)
	for i in range(noise.shape[0]):
		noise[i], bias2[i], variance[i], error[i] = mean_bias_variance(X[:n[i]], y[:n[i]], Ridge(alpha=alpha))

	plt.figure()
	plt.xlabel("$n$")
	plt.ylabel("mean noise")
	plt.grid(True)
	plt.semilogx(n, noise)
	plt.savefig(dirs + "{}.pdf".format("mean_noise_n"), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.xlabel("$n$")
	plt.grid(True)
	plt.loglog(n, bias2, label="mean bias$^2$")
	plt.loglog(n, variance, label="mean variance")
	plt.loglog(n, error, label="mean error")
	plt.legend()
	plt.savefig(dirs + "{}.pdf".format("rrg_mean_n"), bbox_inches='tight')
	plt.close()

	### q

	n = 10 ** 3
	q = np.arange(0, 8)

	noise = np.empty(len(q))
	bias2 = np.empty(noise.shape)
	variance = np.empty(noise.shape)
	error = np.empty(noise.shape)

	X, y = make_data(n, q[-1])
	for i in range(noise.shape[0]):
		noise[i], bias2[i], variance[i], error[i] = mean_bias_variance(X[:,:q[i] + 1], y, Ridge(alpha=alpha))

	plt.figure()
	plt.xlabel("$q$")
	plt.ylabel("mean noise")
	plt.grid(True)
	plt.plot(q, noise)
	plt.savefig(dirs + "{}.pdf".format("mean_noise_q"), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.xlabel("$q$")
	plt.grid(True)
	plt.semilogy(q, bias2, label="mean bias$^2$")
	plt.semilogy(q, variance, label="mean variance")
	plt.semilogy(q, error, label="mean error")
	plt.legend()
	plt.savefig(dirs + "{}.pdf".format("rrg_mean_q"), bbox_inches='tight')
	plt.close()

	### alpha

	q = 0
	alpha = 10 ** np.arange(-6., 3.)

	noise = np.empty(len(alpha))
	bias2 = np.empty(noise.shape)
	variance = np.empty(noise.shape)
	error = np.empty(noise.shape)

	X, y = make_data(n, q)
	for i in range(noise.shape[0]):
		noise[i], bias2[i], variance[i], error[i] = mean_bias_variance(X, y, Ridge(alpha=alpha[i]))

	plt.figure()
	plt.xlabel("$\\alpha$")
	plt.grid(True)
	plt.loglog(alpha, bias2, label="mean bias$^2$")
	plt.loglog(alpha, variance, label="mean variance")
	plt.loglog(alpha, error, label="mean error")
	plt.legend()
	plt.savefig(dirs + "{}.pdf".format("rrg_mean_alpha"), bbox_inches='tight')
	plt.close()

	## K-Neighbors regression

	### n

	n = (10 ** np.arange(2.4, 5.2, 0.2)).astype(int)
	q = 0
	k = 5

	noise = np.empty(len(n))
	bias2 = np.empty(noise.shape)
	variance = np.empty(noise.shape)
	error = np.empty(noise.shape)

	X, y = make_data(n[-1], q)
	for i in range(noise.shape[0]):
		noise[i], bias2[i], variance[i], error[i] = mean_bias_variance(X[:n[i]], y[:n[i]], KNeighborsRegressor(n_neighbors=k))

	plt.figure()
	plt.xlabel("$n$")
	plt.grid(True)
	plt.loglog(n, bias2, label="mean bias$^2$")
	plt.loglog(n, variance, label="mean variance")
	plt.loglog(n, error, label="mean error")
	plt.legend()
	plt.savefig(dirs + "{}.pdf".format("knr_mean_n"), bbox_inches='tight')
	plt.close()

	### q

	n = 10 ** 3
	q = np.arange(0, 8)

	noise = np.empty(len(q))
	bias2 = np.empty(noise.shape)
	variance = np.empty(noise.shape)
	error = np.empty(noise.shape)

	X, y = make_data(n, q[-1])
	for i in range(noise.shape[0]):
		noise[i], bias2[i], variance[i], error[i] = mean_bias_variance(X[:,:q[i] + 1], y, KNeighborsRegressor(n_neighbors=k))

	plt.figure()
	plt.xlabel("$q$")
	plt.grid(True)
	plt.semilogy(q, bias2, label="mean bias$^2$")
	plt.semilogy(q, variance, label="mean variance")
	plt.semilogy(q, error, label="mean error")
	plt.legend()
	plt.savefig(dirs + "{}.pdf".format("knr_mean_q"), bbox_inches='tight')
	plt.close()

	### k

	q = 0
	k = np.arange(1, 31)

	noise = np.empty(len(k))
	bias2 = np.empty(noise.shape)
	variance = np.empty(noise.shape)
	error = np.empty(noise.shape)

	X, y = make_data(n, q)
	for i in range(noise.shape[0]):
		noise[i], bias2[i], variance[i], error[i] = mean_bias_variance(X, y, KNeighborsRegressor(n_neighbors=k[i]))

	plt.figure()
	plt.xlabel("$k$")
	plt.grid(True)
	plt.semilogy(k, bias2, label="mean bias$^2$")
	plt.semilogy(k, variance, label="mean variance")
	plt.semilogy(k, error, label="mean error")
	plt.legend()
	plt.savefig(dirs + "{}.pdf".format("knr_mean_k"), bbox_inches='tight')
	plt.close()
