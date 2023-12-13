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

with open('../../settings.json', 'r') as json_file:
    settings = json.load(json_file)

dataset_name = settings.get('dataset_name')
split_train_ratio = settings.get('split_train_ratio')
split_test_ratio = 1 - split_train_ratio
rndm_state = settings.get('rndm_state')

output_dir = "./CICIDS-hypertune"
train_path = "../../input/CICIDS/small-CICIDS.csv"
test_path = "../../input/CICIDS/small-CICIDS.csv"
Path(output_dir).mkdir(parents=True, exist_ok=True)
# Read Train and Test datasets
data_train = pd.read_csv(train_path)
data_train.columns = data_train.columns.str.strip().tolist()

# Read dataset configuration from JSON
with open('../../dataset-config.json', 'r') as file:
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

pca = PCA(n_components=feature_reduced_number)
pca = pca.fit(x)
x_reduced = pca.transform(x)
y = y.astype('int')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_test_ratio, random_state=rndm_state)
x_train_reduced, x_test_reduced, y_train_reduced, y_test_reduced = train_test_split(x_reduced, y, test_size=split_test_ratio, random_state=rndm_state)

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

    elif model_name == "svm":
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
study_lr = optuna.create_study(direction='maximize')
study_lr.optimize(lambda trial: objective(trial, "lr"), n_trials=100)
best_params_lr = study_lr.best_params

# Optimize hyperparameters for K-Nearest Neighbors
study_knn = optuna.create_study(direction='maximize')
study_knn.optimize(lambda trial: objective(trial, "knn"), n_trials=100)
best_params_knn = study_knn.best_params

# Optimize hyperparameters for XGBoost
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(lambda trial: objective(trial, "xgb"), n_trials=100)
best_params_xgb = study_xgb.best_params

# Optimize hyperparameters for GaussianNB
study_gnb = optuna.create_study(direction='maximize')
study_gnb.optimize(lambda trial: objective(trial, "gnb"), n_trials=100)
best_params_gnb = study_gnb.best_params

# Optimize hyperparameters for LinearSVC
study_lsvc = optuna.create_study(direction='maximize')
study_lsvc.optimize(lambda trial: objective(trial, "svm"), n_trials=100)
best_params_lsvc = study_lsvc.best_params

# Optimize hyperparameters for Decision Trees
study_dt = optuna.create_study(direction='maximize')
study_dt.optimize(lambda trial: objective(trial, "dt"), n_trials=100)
best_params_dt = study_dt.best_params

# Optimize hyperparameters for RandomForestClassifier
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(lambda trial: objective(trial, "rf"), n_trials=100)
best_params_rf = study_rf.best_params

# Print the best hyperparameters
print("Best Hyperparameters for Logistic Regression:", best_params_lr)
print("Best Hyperparameters for K-Nearest Neighbors:", best_params_knn)
print("Best Hyperparameters for XGBoost:", best_params_xgb)
print("Best Hyperparameters for GaussianNB:", best_params_gnb)
print("Best Hyperparameters for LinearSVC:", best_params_lsvc)
print("Best Hyperparameters for DecisionTree:", best_params_dt)
print("Best Hyperparameters for RandomForestClassifier:", best_params_rf)

# Save results to a file
best_params = {
    "Logistic Regression": best_params_lr,
    "K-Nearest Neighbors": best_params_knn,
    "XGBoost": best_params_xgb,
    "GaussianNB": best_params_gnb,
    "LinearSVC": best_params_lsvc,
    "DecisionTree": best_params_dt,
    "RandomForestClassifier": best_params_rf
}

with open(f"{output_dir}/UNSW-NB15_best_hyperparameters.json", "w") as f:
    json.dump(best_params, f, indent=2)

print("Best hyperparameters saved to 'best_hyperparameters.json'")