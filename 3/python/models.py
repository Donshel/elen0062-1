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
import itertools as itt

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.svm import SVC


###########
# Classes #
###########

class DTC(DecisionTreeClassifier):
    pass


class KNN(KNeighborsClassifier):
    pass


class LDA(LinearDiscriminantAnalysis):
    pass


class MLP(MLPClassifier):
    pass


class RFC(RandomForestClassifier):
    pass


class SVM(SVC): # probability=True
    pass


class VC(VotingClassifier): # voting='soft'
    pass


class SC(StackingClassifier): # use_probas=True, average_probas=False
    pass


class ConsensusClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, models, majority=None):
        self.models = models

        if majority is None:
            self.majority = len(self.models) // 2 + len(self.models) % 2
        else:
            self.majority = majority

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

        return self

    def concensus(self, P):
        n = P.shape[1]
        C = np.zeros((P.shape[0], 1))

        for k in range(self.majority, n + 1):
            for index in itt.combinations(range(n), k):
                i_in = list(index)
                i_out = [i for i in range(n) if i not in i_in]
                C[:, 0] += P[:, i_in].prod(axis=1) * (1 - P[:, i_out]).prod(axis=1)

        return C

    def predict_proba(self, X):
        proba = np.zeros((X.shape[0], len(self.models)))

        for i, model in enumerate(self.models):
            proba[:, i] = model.predict_proba(X)[:, -1]

        proba = self.concensus(proba)
        proba = np.hstack([1 - proba, proba])

        return proba

    def predict(self, X):
        y = np.argmax(self.predict_proba(X), axis=1)

        return y
