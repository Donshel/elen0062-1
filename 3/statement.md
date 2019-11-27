# Activity prediction for chemical compounds

See the [IML2019](https://www.kaggle.com/c/iml2019/) kaggle competition.

## Description

This competition is organized in the context of the *Introduction to Machine Learning* course (ELEN0062-1) at the University of Liège, Belgium. The goal of the competition is to let you apply the methods and principles exposed in the theoretical course in order to address a real problem: activity prediction for chemical compounds.

### Task

The mission of your group, should you decide to accept it, is to use (supervised) learning techniques to design a model able to predict the activity of a chemical compound (described by its molecular structure).

To achieve this, we provide you with a set of `15939` chemical compounds with their corresponding activities. A test set of `9563` unlabelled chemical compounds is provided for which we ask you to provide an activity prediction.

Note that you cannot use any information outside of those provided.

### Problem description

The original study was about determining the ability for a chemical compound to inhibit HIV replication. Therefore, tens of thousands of compounds have been checked for evidence of anti-HIV activity.
The dataset is made of two components:

* chemical structural data on compounds: each chemical compound is described under the SMILES format. SMILES, standing for *Simplified Molecular Input Line Entry Specification*, is a line notation for the chemical structure of molecules.
* HIV-activity : it corresponds to the screening result evaluating the activity (`1`) or the inactivity (`0`) of the chemical compound.

### Toolkit for cheminformatics RDKit

In order to generate features from SMILES, you may employ the open source toolkit for cheminformatics RDKit, see [RDKit Documentation](https://rdkit.readthedocs.io/).

In particular, you may need the installation steps, see [How to install RDKit](https://rdkit.readthedocs.io/en/latest/Install.html#how-to-install-rdkit-with-conda).

An example of how to generate features with this toolkit is given in the `toy_example.py` script.

## Evaluation

### Evaluation metrics

* Task 1 - activity prediction for chemical compounds: The evaluation metric is the ROC AUC (area under the ROC curve) as seen in the course.

* Task 2 - estimate your model performance : The evaluation is made as follows
	```
	AUC_ts - |AUC_pred - AUC_ts|
	```
	where `AUC_ts` is your actual score on the testing set (i.e., for task 1) and `AUC_pred` is your estimate of the AUC.

##### Important remarks !

* During the competition, the ranking only takes into account the ROC AUC metric (task 1) and this AUC is estimated only from a subset of the 9562 test examples (roughly, one third of this set).
* After the competition, your final submission will be evaluated on both tasks (tasks 1 & 2), on the part of the test set (two thirds) which was not used during the competition.
* Since the classification problem is very unbalanced, AUC estimates from one third of the test set are likely not to be very reliable. Be careful not to overfit this set!

### Submission format

A sample submission file is provided as an illustration. Submission files should contain two columns: `Chem_ID` and `Prediction`. `Chem_ID` represents the ID of a test chemical compound, and `Prediction` is the prediction of your model associated to that ID.

Please note that `CHEM_ID = Chem_0` corresponds to your estimate of the AUC of your model.

The file should contain a header and have the following format :

```csv
"Chem_ID","Prediction"
Chem_0,0.50
Chem_1,0.10
Chem_2,0.98
...
...
Chem_12750,0.24
```

where the estimated AUC is 0.50 and the predictions start on row 3; the predicted probability that the first chemical compound of the test set is labelled with ACTIVE = 1 is 0.10 in this example.

## Data

```bash
kaggle competitions download -c iml2019
```

### File description

* `learning_set.csv` - the training set;
* `test_set.csv` - the test set;
* `toy_example.py` - toy script that helps you to make your first submission.

### Data fields

* `SMILES` - *Simplified Molecular Input Line Entry Specification* (SMILES) representation of chemical compound;
* `ACTIVE` - target binary variable specifying if the chemical compound is active or not;
* `CHEM_ID` - a unique id for test samples (not in the training set).

### `toy_example.py`

This *naive* script helps you to start by :

* Loading training and test sets;
* Deriving features (fingerprints) from chemical structural data (under the inline description SMILES);
* Making random predictions (and using a decision tree if uncommented).
