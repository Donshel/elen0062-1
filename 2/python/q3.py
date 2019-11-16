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
	def make_data(N, q):
		X = x_dom[choice(len(x_dom), (N, q + 1))]
		y = f(X[:, 0]).reshape(N, 1) + sigma * randn(N, 1)

		return X, y

	# 3(d) Estimations

	def bias_variance(X, y, model, p=20):
		N = X.shape[0]
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
		index = np.arange(N // p)
		for i in range(p):
			model.fit(X[index, :], y[index])
			y_hat[:, i] = model.predict(dom).reshape(len(dom))

			index += N // p

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

	np.random.seed(0)

	N = 10 ** 4
	q = 0

	X, y = make_data(N, q)

	### mkdir -p products/pdf
	dirs = "products/pdf/"
	os.makedirs(dirs, exist_ok=True)

	plt.figure()
	plt.xlabel("$x_r$")
	plt.ylabel("$y$")
	plt.grid(True)
	plt.scatter(X[:100], y[:100], color="g", label="$LS$ subset")
	#plt.plot(x_dom, f(x_dom), label="$f(x_r)$")
	plt.plot(x_dom, Ridge().fit(X, y).predict(x_dom.reshape(200, 1)), label="Ridge")
	plt.plot(x_dom, KNeighborsRegressor().fit(X, y).predict(x_dom.reshape(200, 1)), label="K-Neighbors")
	plt.legend()
	plt.savefig(dirs + "{}.pdf".format("predictions"), bbox_inches='tight')
	plt.close()

	## Ridge regression

	dom, noise, bias2, variance, error = bias_variance(X, y, Ridge())

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

	dom, noise, bias2, variance, error = bias_variance(X, y, KNeighborsRegressor())

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

	### N

	N = (10 ** np.arange(2, 5.2, 0.2)).astype(int)
	q = 0
	alpha = 1.0

	noise = np.empty(len(N))
	bias2 = np.empty(noise.shape)
	variance = np.empty(noise.shape)
	error = np.empty(noise.shape)

	X, y = make_data(N[-1], q)
	for i in range(noise.shape[0]):
		noise[i], bias2[i], variance[i], error[i] = mean_bias_variance(X[:N[i]], y[:N[i]], Ridge(alpha=alpha))

	plt.figure()
	plt.xlabel("$N$")
	plt.ylabel("mean noise")
	plt.grid(True)
	plt.semilogx(N, noise)
	plt.savefig(dirs + "{}.pdf".format("mean_noise_n"), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.xlabel("$N$")
	plt.grid(True)
	plt.loglog(N, bias2, label="mean bias$^2$")
	plt.loglog(N, variance, label="mean variance")
	plt.loglog(N, error, label="mean error")
	plt.legend()
	plt.savefig(dirs + "{}.pdf".format("rrg_mean_n"), bbox_inches='tight')
	plt.close()

	### q

	N = 10 ** 3
	q = np.arange(0, 8)

	noise = np.empty(len(q))
	bias2 = np.empty(noise.shape)
	variance = np.empty(noise.shape)
	error = np.empty(noise.shape)

	X, y = make_data(N, q[-1])
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

	X, y = make_data(N, q)
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

	### N

	N = (10 ** np.arange(2, 5.2, 0.2)).astype(int)
	q = 0
	k = 5

	noise = np.empty(len(N))
	bias2 = np.empty(noise.shape)
	variance = np.empty(noise.shape)
	error = np.empty(noise.shape)

	X, y = make_data(N[-1], q)
	for i in range(noise.shape[0]):
		noise[i], bias2[i], variance[i], error[i] = mean_bias_variance(X[:N[i]], y[:N[i]], KNeighborsRegressor(n_neighbors=k))

	plt.figure()
	plt.xlabel("$N$")
	plt.grid(True)
	plt.loglog(N, bias2, label="mean bias$^2$")
	plt.loglog(N, variance, label="mean variance")
	plt.loglog(N, error, label="mean error")
	plt.legend()
	plt.savefig(dirs + "{}.pdf".format("knr_mean_n"), bbox_inches='tight')
	plt.close()

	### q

	N = 10 ** 3
	q = np.arange(0, 8)

	noise = np.empty(len(q))
	bias2 = np.empty(noise.shape)
	variance = np.empty(noise.shape)
	error = np.empty(noise.shape)

	X, y = make_data(N, q[-1])
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
	k = np.arange(1, 51)

	noise = np.empty(len(k))
	bias2 = np.empty(noise.shape)
	variance = np.empty(noise.shape)
	error = np.empty(noise.shape)

	X, y = make_data(N, q)
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
