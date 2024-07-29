# Intrusion Detection System with Machine Learning
This repository contains the code for the project "Development and Implementation of Machine Learning-Based Anomaly Detection System". This repository proposes a Python implementation of an Intrusion Detection System (IDS) using machine learning models. The system is designed to identify and classify network intrusions or attacks, providing a robust defense against potential security threats. This project modifies the work from Essam Mohamed to train and test models using CICIDS2017 dataset instead.

This project makes use of several popular machine learning models, including Logistic regression, K-nearest neighbours, Gaussian Naive Bayes, Linear SVC, Decision trees, Extra Trees, XGBoost, Random forest, and Deep Neural Network. Hyperparameter tuning is done via Optuna for each models. Feature extraction is done by seelcting the top 40 features from tree-based models (Decision Trees, Extra Trees, XGboost, Random Forest).

# Prerequisites
Some things need to be done before you can run the program:

## Python
This project is built on Python 3.10. For the dependencies, please see `requirements.txt`.

## Dataset
The code takes CICIDS2017 dataset. The location of the dataset should be defined in `dataset-config.json`. This repository does not include the dataset itself, the dataset can be obtained from [UNB 's official website](https://www.unb.ca/cic/datasets/ids-2017.html). 

# Dataset Configuration
This project uses a dataset configuration file (`dataset-config.json`) to specify settings for each dataset. 

## Fields

`train_path` and `test_path`: Paths to the training and testing datasets for the specified dataset.

`read_cols_from_csv`: Read column headers from CSV file.

`cat_cols`: A list of categorical columns in the dataset. These columns may require special handling during preprocessing.

`obj_cols`: A list of object columns to be processed using one-hot encoding.

`drop_cols`: A list of columns to be dropped from the dataset. Useful for removing unnecessary or redundant features.

`label_header`: The column that contains the labels or targets for the machine learning model.

`label_normal_value`: The value in the label column that represents the "normal/benign" class. This is important for classification tasks.

`pie_stats`: Select columns for which pie chart will be generated. Can be used for visualizing data distribution aspects.

`reduced_features`: Specify a list of columns/features to be used when training models, columns/features not specified here will be discarded or ignored. Leave empty to use all columns/features. 

`class_mapping`: Map multi-class label to a fixed number to an integer starting from 0. 

`target_names`: Label to show when displaying confusion matrix and performance metrics.


## Example

```json
{
"CICIDS2017": {
    "train_path": "./input/CICIDS2017.csv",
    "test_path": "./input/LAN8h-combined.csv",
    "read_cols_from_csv": true,
    "cat_cols": ["Label"],
    "drop_cols": [],
    "obj_cols": [],
    "label_header": "Label",
    "label_normal_value": "BENIGN",
    "pie_stats": [["Label", "Label"]],
    "reduced_features" : ["ACK Flag Count", "Average Packet Size", "Avg Bwd Segment Size"],
    "class_mapping": {
      "BENIGN": 0,
      "Bot": 1,
      "DoS": 2,
      "BruteForce": 3,
      "PortScan": 4,
      "Web Attack": 5,
      "Others": 6
    },
    "target_names": ["BENIGN", "Bot", "DoS", "BruteForce", "PortScan", "Web Attack", "Others"]
  },
}
```

# IDS Configuration
The IDS uses a configuration file (`settings.json`) to specify settings for the IDS itself. 

## Fields
  `display_results`: (true/false) Display results/images (such as confusion matrix) immediately after evaluation.

  `generate_statistics_pie`: (true/false) Generate a pie chart for classes in dataset.
  
  `dataset_name`: (string) Specify which entry in dataset-config to use.
  
  `output_dir`: (string) Output directory.
  
  `load_saved_models`: (true/false) Use saved models.
  
  `save_trained_models`: (true/false) Save the trained models in training mode.
  
  `model_save_path`: (string) Model save path.
  
  `model_save_version`: (string) Model save version. This creates a subfolder with specified name under `model_save_path`.
  
  `average`: ("micro", "macro", "weighted") Specify which type of average to use in evaluation metrics.
  
  `bool_gnb`: (true/false) Enable/disable Gaussian NB model.
  
  `bool_xgb`: (true/false) Enable/disable XGBoost model.
  
  `bool_et`: (true/false) Enable/disable Extra Trees model.
  
  `bool_dt`: (true/false) Enable/disable Decision Trees model.
  
  `bool_rf`: (true/false) Enable/disable Random Forest model.
  
  `bool_lr`: (true/false) Enable/disable Logistic Regression model.
  
  `bool_knn`: (true/false) Enable/disable KNN model.
  
  `bool_lin_svc`: (true/false) Enable/disable Linear SVC model.
  
  `bool_dnn`: (true/false) Enable/disable DNN model.
  
  `use_single_dataset`: (true/false) When set to true, splits a single dataset into train and test sets. When set to false, uses two different datasets specified in dataset-config to train and test the models.
  
  `enable_SMOTE`: (true/false) Enable/disable SMOTE oversampling when training models.
  
  `scaler`: ("StandardScaler", RobustScaler, "") Scaling/normalization of training dataset. Set to empty to skip scaling step. 
  
  `split_train_ratio`: (float, 0.0 to 1.0) Sets train_test_split training ratio.
  
  `split_test_ratio`: (float, 0.0 to 1.0) Sets train_test_split testing ratio.
  
  `rndm_state`: (int) Sets random state for models.
  
  `max_workers`: (int) Sets maximum amount of models allowed to run simultaneously. KNN and XGBoost is not affected by this setting.
  
  `delay_start`: (int) Delay start the script in seconds.


## Example
```json
{
  "display_results": false,
  "generate_statistics_pie": true,
  "dataset_name": "CICIDS2017",
  "output_dir": "./output/CICIDS2017",
  "load_saved_models": false,
  "save_trained_models": true,
  "model_save_path": "./Saved models/",
  "model_save_version": "v1.0",
  "average": "weighted",
  "bool_gnb": true,
  "bool_xgb": true,
  "bool_et": true,
  "bool_dt": true,
  "bool_rf": true,
  "bool_lr": true,
  "bool_knn": true,
  "bool_lin_svc": true,
  "bool_dnn": true,
  "use_single_dataset": true,
  "enable_SMOTE": true,
  "scaler":"StandardScaler",
  "split_train_ratio": 0.6,
  "split_test_ratio": 0.4,
  "rndm_state": 42,
  "max_workers": 6,
  "delay_start": 0
}
```


# Credits
This codebase is based on the work of Essam Mohamed, available on Kaggle. The original code focused on building an Intrusion Detection System using machine learning and deep learning techniques, using the NSL-KDD dataset.

Original Author: Essam Mohamed

Kaggle Profile: [Essam Mohamed on Kaggle](https://www.kaggle.com/code/essammohamed4320)

Link to Original Code: [Intrusion Detection System with ML/DL](https://www.kaggle.com/code/essammohamed4320/intrusion-detection-system-with-ml-dl)
