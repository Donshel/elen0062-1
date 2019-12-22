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
import time

import pandas as pd
import numpy as np


#############
# Functions #
#############

def load_from_csv(path, delimiter=','):
    """Load csv file and return a NumPy array of its data.

    Inputs
    ------
    path : str
        the path to the csv file to load
    delimiter : str (default : ',')
        the csv field delimiter

    Output
    ------
    D : array
        the NumPy array of the data contained in the file
    """

    if not os.path.exists(path):
        raise FileNotFoundError('File "{}" does not exists.'.format(path))

    return pd.read_csv(path, delimiter=delimiter)


def make_submission(y_predicted, auc_predicted, file_name='submission', date=True, indexes=None):
    """Write a submission file for the Kaggle platform.

    Inputs
    ------
    y_predicted : array [n_predictions, 1]
        if 'y_predict[i]'' is the prediction
        for chemical compound 'i' (or indexes[i] if given)

    auc_predicted : float [1]
        the estimated ROCAUC of y_predicted

    file_name : str or None (default: 'submission')
        The path to the submission file to create (or override). If none is
        provided, a default one will be used. Also note that the file extension
        (.txt) will be appended to the file.

    date : boolean (default: True)
        whether to append the date in the file name

    Output
    ------
    file_name : path
        the final path to the submission file
    """

    # Naming the file
    if date:
        file_name = '{}_{}'.format(file_name, time.strftime('%d-%m-%Y_%Hh%M'))

    file_name = '{}.txt'.format(file_name)

    # Creating default indexes if not given
    if indexes is None:
        indexes = np.arange(len(y_predicted)) + 1

    # Writing into the file
    with open(file_name, 'w') as handle:
        handle.write('"Chem_ID","Prediction"\n')
        handle.write('Chem_{:d},{}\n'.format(0, auc_predicted))

        for n, idx in enumerate(indexes):
            if np.isnan(y_predicted[n]):
                raise ValueError('The prediction cannot be NaN')

            line = 'Chem_{:d},{}\n'.format(idx, y_predicted[n])
            handle.write(line)

    return file_name
