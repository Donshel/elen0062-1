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

import os
import utils

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score, ShuffleSplit

from models import DTC, KNN, LDA, MLP, RFC, SVM
from models import VC, SC, ConsensusClassifier

import fingerprints 
import kernels


##############
# Parameters #
##############

# Paths to training and testing set
TRAINING_SET = '../resources/csv/training_set.csv'
TEST_SET = '../resources/csv/test_set.csv'

# Path to export predictions
DESTINATION = '../products/'

# Fingerprint transformation
FINGERPRINT = fingerprints.morgan()

# Model to train
MODEL = ConsensusClassifier([KNN(n_neighbors=17), MLP(random_state=0), SVM(gamma='auto', random_state=0, probability=True), RFC(500, random_state=0)])



########
# Main #
########

if __name__ == '__main__':
    # Load training and test set
    LS = utils.load_from_csv(TRAINING_SET)
    TS = utils.load_from_csv(TEST_SET)

    # Create fingerprint features and output of learning set
    X_LS = fingerprints.transform(LS['SMILES'].values, FINGERPRINT)
    y_LS = LS['ACTIVE'].values

    # Variance threshold (feature selection)
    selector = VarianceThreshold()
    selector.fit(X_LS)
    X_LS = selector.transform(X_LS)

    # Cross validation score
    cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
    scores = cross_val_score(MODEL, X_LS, y_LS, cv=cv, scoring='roc_auc')

    # Estimated AUC
    AUC = scores.mean()

    # Train model
    MODEL.fit(X_LS, y_LS)

    # Create fingerprint features of test set
    X_TS = fingerprints.transform(TS['SMILES'].values, FINGERPRINT)
    X_TS = selector.transform(X_TS)

    # Predict
    prob = MODEL.predict_proba(X_TS)[:, -1]

    # Writing the submission file
    os.makedirs(DESTINATION, exist_ok=True)
    fname = utils.make_submission(prob, AUC, DESTINATION + 'submission')

    print('Submission file "{}" successfully written'.format(fname))
