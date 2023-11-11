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
from pathlib import Path
import optuna

output_dir = "./UNSW-NB15-hypertune"
train_path = "../../input/UNSW_NB15/UNSW_NB15_training-set.csv"
test_path = "../../input/UNSW_NB15/UNSW_NB15_testing-set.csv"
Path(output_dir).mkdir(parents=True, exist_ok=True)
# Read Train and Test datasets
data_train = pd.read_csv(train_path)
data_test = pd.read_csv(test_path)

columns = (['id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'attack_cat', 'label'])

data_train.columns = columns
data_test.columns = columns

print("Mapping outcomes...", flush=True)
data_train.loc[data_train['label'] == "Normal", "label"] = 0
data_train.loc[data_train['label'] != 0, "label"] = 1

data_test.loc[data_test['label'] == "Normal", "label"] = 0
data_test.loc[data_test['label'] != 0, "label"] = 1

def Scaling(df_num, cols):
    std_scaler = RobustScaler()
    std_scaler_temp = std_scaler.fit_transform(df_num)
    std_df = pd.DataFrame(std_scaler_temp, columns=cols)
    return std_df

def preprocess(dataframe):
    df_num = dataframe.drop(cat_cols, axis=1)
    num_cols = df_num.select_dtypes(include=[np.number]).columns
    obj_cols = ['proto', 'service', 'state']
    dataframe = pd.get_dummies(dataframe, columns=obj_cols)
    df_num = dataframe[num_cols]
    labels = dataframe['label']
    std_scaler = RobustScaler()
    std_scaler_temp = std_scaler.fit_transform(df_num)
    std_df = pd.DataFrame(std_scaler_temp, columns=num_cols)
    dataframe = pd.concat([std_df, labels], axis=1)
    return dataframe

cat_cols = ['attack_cat', 'label']
scaled_train = preprocess(data_train)
scaled_test = preprocess(data_test)

x_train = scaled_train.drop(['label'], axis=1).values
y_train = scaled_train['label'].values

x_test = scaled_test.drop(['label'], axis=1).values
y_test = scaled_test['label'].values

y_train = y_train.astype('int')
y_test = y_test.astype('int')


def objective(trial, model_name):

    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 2, 50),
    }
    model = KNeighborsClassifier(**params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy

study_knn = optuna.create_study(direction='maximize')
study_knn.optimize(lambda trial: objective(trial, "knn"), n_trials=50)
best_params_knn = study_knn.best_params

print("Best Hyperparameters for K-Nearest Neighbors:", best_params_knn)

# Save results to a file
best_params = {
    "K-Nearest Neighbors": best_params_knn
}

with open(f"{output_dir}/UNSW-NB15_knn_hyperparameters.json", "w") as f:
    json.dump(best_params, f, indent=2)

print("Saved results'")