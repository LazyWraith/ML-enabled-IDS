import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import optuna

###--------SETTINGS--------###
#
display_results = False
#
##############################

# result output
output_dir = "./UNSW-NB15-hypertune"
train_path = "../input/UNSW_NB15/UNSW_NB15_training-set.csv"

# Read Train and Test dataset
data_train = pd.read_csv(train_path)

columns = (['id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'attack_cat', 'label'])

# Assign names for columns
data_train.columns = columns
# data_train.info()
# data_train.describe().style.background_gradient(cmap='Blues').set_properties(**{'font-family': 'Segoe UI'})

print("Mapping outcomes...", flush=True)
data_train.loc[data_train['label'] == "Normal", "label"] = 0
data_train.loc[data_train['label'] != 0, "label"] = 1

def Scaling(df_num, cols):
    std_scaler = RobustScaler()
    std_scaler_temp = std_scaler.fit_transform(df_num)
    std_df = pd.DataFrame(std_scaler_temp, columns=cols)
    return std_df

def preprocess(dataframe):
    print("Running preprocess...", flush=True)
    df_num = dataframe.drop(cat_cols, axis=1)
    num_cols = df_num.columns
    scaled_df = Scaling(df_num, num_cols)

    dataframe.drop(labels=num_cols, axis="columns", inplace=True)
    dataframe[num_cols] = scaled_df[num_cols]

    dataframe.loc[dataframe['label'] == "Normal", "label"] = 0
    dataframe.loc[dataframe['label'] != 0, "label"] = 1

    dataframe = pd.get_dummies(dataframe, columns=['proto', 'service', 'state', 'attack_cat'])
    return dataframe

cat_cols = ['proto', 'service', 'state', 'attack_cat']
scaled_train = preprocess(data_train)

x = scaled_train.drop(['label'] , axis = 1).values
y = scaled_train['label'].values

# pca = PCA(n_components=20)
# pca = pca.fit(x)
# x_reduced = pca.transform(x)
# print(f"Number of original features is {x.shape[1]} and of reduced features is {x_reduced.shape[1]}")

y = y.astype('int')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# x_train_reduced, x_test_reduced, y_train_reduced, y_test_reduced = train_test_split(x_reduced, y, test_size=0.2, random_state=42)


def objective(trial, model_name):
    if model_name == "lr":
        params = {
            'C': trial.suggest_float('C', 0.001, 10, log=True),
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear']),
        }
        model = LogisticRegression(**params)

    elif model_name == "knn":
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 50),
        }
        model = KNeighborsClassifier(**params)

    elif model_name == "xgb":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            # 'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        }
        model = xgb.XGBClassifier(**params)

    elif model_name == "gnb":
        params = {
            'var_smoothing': trial.suggest_float('var_smoothing', 1e-12, 1e-2, log=True),
        }
        model = GaussianNB(**params)

    elif model_name == "lsvc":
        params = {
            'C': trial.suggest_float('C', 1e-5, 1e5, log=True),
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
        }
        model = svm.LinearSVC(**params)

    elif model_name == "dt":
        params = {
            'max_depth' : trial.suggest_int('max_depth', 3, 15),
            'min_samples_split' : trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 10),
        }
        model = DecisionTreeClassifier(**params)
    
    elif model_name == "rf":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        }
        model = RandomForestClassifier(**params)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy

# Optimize hyperparameters for Logistic Regression
# study_lr = optuna.create_study(direction='maximize')
# study_lr.optimize(lambda trial: objective(trial, "lr"), n_trials=100)
# best_params_lr = study_lr.best_params

# # Optimize hyperparameters for K-Nearest Neighbors
# study_knn = optuna.create_study(direction='maximize')
# study_knn.optimize(lambda trial: objective(trial, "knn"), n_trials=100)
# best_params_knn = study_knn.best_params

# # Optimize hyperparameters for XGBoost
# study_xgb = optuna.create_study(direction='maximize')
# study_xgb.optimize(lambda trial: objective(trial, "xgb"), n_trials=100)
# best_params_xgb = study_xgb.best_params

# # Optimize hyperparameters for GaussianNB
# study_gnb = optuna.create_study(direction='maximize')
# study_gnb.optimize(lambda trial: objective(trial, "gnb"), n_trials=100)
# best_params_gnb = study_gnb.best_params

# # Optimize hyperparameters for LinearSVC
study_lsvc = optuna.create_study(direction='maximize')
study_lsvc.optimize(lambda trial: objective(trial, "lsvc"), n_trials=100)
best_params_lsvc = study_lsvc.best_params

# # Optimize hyperparameters for Decision Trees
# study_dt = optuna.create_study(direction='maximize')
# study_dt.optimize(lambda trial: objective(trial, "dt"), n_trials=100)
# best_params_dt = study_dt.best_params

# # Optimize hyperparameters for RandomForestClassifier
# study_rf = optuna.create_study(direction='maximize')
# study_rf.optimize(lambda trial: objective(trial, "rf"), n_trials=100)
# best_params_rf = study_rf.best_params

# Print the best hyperparameters
# print("Best Hyperparameters for Logistic Regression:", best_params_lr)
# print("Best Hyperparameters for K-Nearest Neighbors:", best_params_knn)
# print("Best Hyperparameters for XGBoost:", best_params_xgb)
# print("Best Hyperparameters for GaussianNB:", best_params_gnb)
print("Best Hyperparameters for LinearSVC:", best_params_lsvc)
# print("Best Hyperparameters for DecisionTree:", best_params_dt)
# print("Best Hyperparameters for RandomForestClassifier:", best_params_rf)

# Save results to a file
best_params = {
    # "Logistic Regression": best_params_lr,
    # "K-Nearest Neighbors": best_params_knn,
    # "XGBoost": best_params_xgb,
    # "GaussianNB": best_params_gnb,
    "LinearSVC": best_params_lsvc,
    # "DecisionTree": best_params_dt,
    # "RandomForestClassifier": best_params_rf
}

with open(f"{output_dir}/UNSW-NB15_lsvc_hyperparameters.json", "w") as f:
    json.dump(best_params, f, indent=2)

print("Saved results'")