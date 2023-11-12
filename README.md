# Intrusion Detection System with ML
This repository contains the code for the project "Development and Implementation of Machine Learning-Based Anomaly Detection System". This repository proposes a Python implementation of an Intrusion Detection System (IDS) using machine learning models. The system is designed to identify and classify network intrusions or attacks, providing a robust defense against potential security threats. This project modifies the work from Essam Mohamed to train models using UNSW-NB15 instead.

This project makes use of several popular machine learning models, including Logistic regression, K-nearest neighbours, Gaussian Naive Bayes, Linear SVC, Decision trees, XGBoost, and Random forest. Hyperparameter tuning is done via Optuna for each models. Feature extraction is done using PCA, which reduces the dataset features to 20. Currently, feature extraction is only done on XGBoost and Random forest.

# Prerequisites
Some things need to be done before you can run the program:

## Python
This project is built on Python 3.10. For the dependencies, please see `requirements.txt`.

## Dataset
The code takes both UNSW-NB15's training and testing datasets. The datasets should be placed in `./input/UNSW-NB15`. This repository does not include the dataset itself, the dataset can be obtained from [UNSW Sydney's official website](https://research.unsw.edu.au/projects/unsw-nb15-dataset). A partition from this dataset was configured as a training set and testing set, namely, UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv respectively. The number of records in the training set is 175,341 records and the testing set is 82,332 records from the different types, attack and normal.

# Credits
This codebase is based on the work of Essam Mohamed, available on Kaggle. The original code focused on building an Intrusion Detection System using machine learning and deep learning techniques, using the NSL-KDD dataset.

Original Author: Essam Mohamed

Kaggle Profile: [Essam Mohamed on Kaggle](https://www.kaggle.com/code/essammohamed4320)

Link to Original Code: [Intrusion Detection System with ML/DL](https://www.kaggle.com/code/essammohamed4320/intrusion-detection-system-with-ml-dl)
