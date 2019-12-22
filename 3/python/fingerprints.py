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

from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from rdkit.Chem.rdmolops import RDKFingerprint, LayeredFingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonFP


###########
# Methods #
###########

def transform(X, f):
    X = [f(Chem.MolFromSmiles(x)) for x in X]
    X = np.array(X)

    return X


def morgan(radius=2, n_bits=128, use_features=True):
    return lambda x: GetMorganFingerprintAsBitVect(x, radius, nBits=n_bits, useFeatures=use_features)


def maccs():
    return lambda x: GenMACCSKeys(x)


def rdk(fpSize=2048):
    return lambda x: RDKFingerprint(x, fpSize=fpSize)


def layer(fpSize=2048):
    return lambda x: LayeredFingerprint(x, fpSize=fpSize)


def avalon(n_bits=2048):
    return lambda x: GetAvalonFP(x, nBits=n_bits)
