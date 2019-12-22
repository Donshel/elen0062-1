"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 3 - Activity prediction for chemical compounds

Authors :
    - Maxime Meurisse
    - Francois Rozet
    - Valentin Vermeylen
"""

#############
# Libraries #
#############

import numpy as np


###########
# Methods #
###########

def base(X, Y):
    x = np.transpose(np.tile(X.sum(axis=1), (Y.shape[0], 1)))
    y = np.tile(Y.sum(axis=1), (X.shape[0], 1))
    z = X.dot(np.transpose(Y))

    return x, y, z


def manhattan(X, Y):
    n = X.shape[1]
    x, y, z = base(X, Y)

    return n - (x + y - 2 * z)


def euclidean(X, Y):
    return np.sqrt(manhattan(X, Y))


def cosine(X, Y):
    x, y, z = base(X, Y)

    return z / np.sqrt(x * y)


def dice(X, Y):
    x, y, z = base(X, Y)

    return 2 * z / (x + y)


def tanimoto(X, Y):
    x, y, z = base(X, Y)

    return z / (x + y - z)
