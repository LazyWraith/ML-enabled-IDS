# Intrusion Detection System with Machine Learning
This repository contains the code for the project "Development and Implementation of Machine Learning-Based Anomaly Detection System". This repository proposes a Python implementation of an Intrusion Detection System (IDS) using machine learning models. The system is designed to identify and classify network intrusions or attacks, providing a robust defense against potential security threats. This project modifies the work from Essam Mohamed to train models using UNSW-NB15 instead.

This project makes use of several popular machine learning models, including Logistic regression, K-nearest neighbours, Gaussian Naive Bayes, Linear SVC, Decision trees, XGBoost, Random forest, and Deep Neural Network. Hyperparameter tuning is done via Optuna for each models. Feature extraction is done using PCA, which reduces the dataset features to 20. Currently, feature extraction is only done on XGBoost and Random forest.

# Prerequisites
Some things need to be done before you can run the program:

## Python
This project is built on Python 3.10. For the dependencies, please see `requirements.txt`.

## Dataset
The code takes both UNSW-NB15's training and testing datasets. The datasets should be placed in `./input/UNSW-NB15`. This repository does not include the dataset itself, the dataset can be obtained from [UNSW Sydney's official website](https://research.unsw.edu.au/projects/unsw-nb15-dataset). A partition from this dataset was configured as a training set and testing set, namely, UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv respectively. The number of records in the training set is 175,341 records and the testing set is 82,332 records from the different types, attack and normal.

# Dataset Configuration

This project uses a dataset configuration file (`dataset-config.json`) to specify settings for each dataset. Here's an explanation of each field:

## Fields

`train_path` and `test_path`: Paths to the training and testing datasets for the specified dataset.

 `cat_cols`: A list of categorical columns in the dataset. These columns may require special handling during preprocessing.

 `obj_cols`: A list of object columns to be processed using one-hot encoding.

 `drop_cols`: A list of columns to be dropped from the dataset. Useful for removing unnecessary or redundant features.

 `label_name_map`: The column that contains the labels or targets for the machine learning model.

 `label_value_map`: The value in the label column that represents the "normal" class. This is important for binary classification tasks.

 `pie_stats`: A list of pairs of columns for which pie charts will be generated. This can be useful for visualizing certain data distribution aspects.

## Example

```json
{
  "UNSW-NB15": {
    "train_path": "./input/UNSW_NB15/UNSW_NB15_training-set.csv",
    "test_path": "./input/UNSW_NB15/UNSW_NB15_testing-set.csv",
    "cat_cols": ["attack_cat", "label"],
    "obj_cols": ["proto", "service", "state"],
    "drop_cols": ["id"],
    "label_name_map": "label",
    "label_value_map": "Normal",
    "pie_stats": [["proto", "service"], ["attack_cat", "label"]]
  }
}
```

# Credits
This codebase is based on the work of Essam Mohamed, available on Kaggle. The original code focused on building an Intrusion Detection System using machine learning and deep learning techniques, using the NSL-KDD dataset.

Original Author: Essam Mohamed

Kaggle Profile: [Essam Mohamed on Kaggle](https://www.kaggle.com/code/essammohamed4320)

Link to Original Code: [Intrusion Detection System with ML/DL](https://www.kaggle.com/code/essammohamed4320/intrusion-detection-system-with-ml-dl)
