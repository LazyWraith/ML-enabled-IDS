import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import json
import optuna

with open('../settings.json', 'r') as json_file:
    settings = json.load(json_file)

dataset_name = settings.get('dataset_name')
split_train_ratio = settings.get('split_train_ratio')
split_test_ratio = 1 - split_train_ratio
rndm_state = settings.get('rndm_state')

output_dir = "./CICIDS-hypertune"
train_path = "../small-CICIDS2017.csv"
test_path = "../small-CICIDS2017.csv"
Path(output_dir).mkdir(parents=True, exist_ok=True)
# Read Train and Test datasets
data_train = pd.read_csv(train_path)
data_train.columns = data_train.columns.str.strip().tolist()

# Read dataset configuration from JSON
with open('../dataset-config.json', 'r') as file:
    datasets_config = json.load(file)

# Check if the dataset name is valid
if dataset_name in datasets_config:
    config = datasets_config[dataset_name]

    # Dataset Path
    train_path = config["train_path"]
    test_path = config["test_path"]

    # Dataset Headers
    read_cols_from_csv = config.get("read_cols_from_csv", True)
    if (not read_cols_from_csv):
        columns = config["columns"]
    cat_cols = config["cat_cols"]
    obj_cols = config["obj_cols"]
    drop_cols = config["drop_cols"]
    label_header = config["label_header"]
    label_normal_value = config["label_normal_value"]
    pie_stats = config["pie_stats"]
    feature_reduced_number = config['feature_reduced_number']
    
else:
    print("Invalid dataset name!")

def preprocess(dataframe, obj_cols_):
    dataframe = dataframe.drop(drop_cols, axis=1)
    df_num = dataframe.drop(cat_cols, axis=1)
    num_cols = df_num.select_dtypes(include=[np.number]).columns
    dataframe = pd.get_dummies(dataframe, columns=obj_cols_)
    df_num = dataframe[num_cols]
    labels = dataframe[label_header]
    std_scaler = RobustScaler()
    std_scaler_temp = std_scaler.fit_transform(df_num)
    std_df = pd.DataFrame(std_scaler_temp, columns=num_cols)
    dataframe = pd.concat([std_df, labels], axis=1)
    return dataframe

print("Mapping outcomes...", flush=True)
data_train.loc[data_train[label_header] == label_normal_value, label_header] = 0
data_train.loc[data_train[label_header] != 0, label_header] = 1

scaled_train = preprocess(data_train, obj_cols)
x = scaled_train.drop(label_header , axis = 1).values
y = scaled_train[label_header].values
y = y.astype('int')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_test_ratio, random_state=rndm_state)

def objective(trial, model_name):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        # 'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
    }
    model = xgb.XGBClassifier(**params)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy

# # Optimize hyperparameters for XGBoost
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(lambda trial: objective(trial, "xgb"), n_trials=1000)
best_params_xgb = study_xgb.best_params

print("Best Hyperparameters for XGBoost:", best_params_xgb)

# Save results to a file
best_params = {
    "XGBoost": best_params_xgb,
}

with open(f"{output_dir}/{dataset_name}_xgb_hyperparameters.json", "w") as f:
    json.dump(best_params, f, indent=2)

print("Saved results'")