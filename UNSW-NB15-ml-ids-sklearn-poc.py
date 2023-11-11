import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import xgboost as xgb
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

###--------SETTINGS--------###
display_results = False
generate_statistics_pie = True
dataset_name = "UNSW-NB15"
output_dir = "./output/UNSW-NB15"
train_path = "./input/UNSW_NB15/UNSW_NB15_training-set.csv"
test_path = "./input/UNSW_NB15/UNSW_NB15_testing-set.csv"
# Models to evaluate
bool_lr         = True
bool_knn        = True
bool_gnb        = True
bool_lin_svc    = True
bool_dt         = True
bool_xgb        = True
bool_rf         = True

###----ML-PARAMETERS-------###

# Logical Regression
lr_params = {
    "C": 0.36585696635446396,
    "max_iter": 868,
    "solver": "lbfgs"
}

# K-Nearest Neighbors
knn_params = {
    "n_neighbors": 21
}

# GaussianNB
gnb_params = {
    "var_smoothing": 9.437310900762216e-08
}

# LinearSVC
lin_svc_params = {
    "C": 1.0000346600564648e-05,
    "max_iter": 709
}

# Decision Trees
dt_params = {
    "max_depth": 10,
    "min_samples_split": 13,
    "min_samples_leaf": 3
}

# XGBoost
xgb_params = {
    "n_estimators": 80,
    "max_depth": 3,
    "learning_rate": 0.001182925709827524
}

# RandomForestClassifier
rf_params = {
    "n_estimators": 87,
    "max_depth": 4,
    "min_samples_split": 9,
    "min_samples_leaf": 9,
    "max_features": 0.793102879894246
}

##############################

Path(output_dir).mkdir(parents=True, exist_ok=True)
# result output
log = open(f'{output_dir}/log.txt', 'w')
filename_counter = 0

def get_filename_counter():
    global filename_counter
    filename_counter += 1
    return str(filename_counter) + ". "

def printlog(message):
    log.write(message + "\n")
    print(message, flush=True)

import time
start_ts = time.time()

def get_ts():
    ts = time.time() - start_ts
    return f"{ts:>08.4f}"

printlog(f"[{get_ts()}] Init complete!")

printlog(f"[{get_ts()}] Reading from {train_path}")

# Read Train and Test dataset
data_train = pd.read_csv(train_path)
data_test = pd.read_csv(test_path)
data_train.head()
columns = (['id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'attack_cat', 'label'])

# Assign names for columns


data_train.columns = columns
data_test.columns = columns
data_train.info()
data_train.describe().style.background_gradient(cmap='Blues').set_properties(**{'font-family': 'Segoe UI'})

printlog(f"[{get_ts()}] Mapping outcomes...")
data_train.loc[data_train['label'] == "Normal", "label"] = 0
data_train.loc[data_train['label'] != 0, "label"] = 1

data_test.loc[data_test['label'] == "Normal", "label"] = 0
data_test.loc[data_test['label'] != 0, "label"] = 1

def pie_plot(df, cols_list, rows, cols):
    # print(f"[{get_ts()}] Generating results...", flush=True)
    fig, axes = plt.subplots(rows, cols)
    for ax, col in zip(axes.ravel(), cols_list):
        df[col].value_counts().plot(ax=ax, kind='pie', figsize=(15, 15), fontsize=10, autopct='%1.0f%%')
        ax.set_title(str(col), fontsize=12)
    counter = get_filename_counter()
    plt.savefig(os.path.join(output_dir, f"{counter}{col}_pie_chart.png"))  # Save the chart
    print(f"[{get_ts()}] Saved results to {output_dir}/{counter}{col}_pie_chart.png", flush=True)
    if (display_results): plt.show()

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


if (generate_statistics_pie):
    pie_plot(data_train, ['proto', 'service'], 1, 2)
    pie_plot(data_train, ['attack_cat', 'label'], 1, 2)

cat_cols = ['attack_cat', 'label']
# Process training set
scaled_train = preprocess(data_train)
x_train = scaled_train.drop(['label'], axis=1).values
y_train = scaled_train['label'].values
y_train = y_train.astype('int')

pca_train = PCA(n_components=20)
x_train_reduced = pca_train.fit_transform(x_train)
y_train_reduced = y_train

# Process testing set
scaled_test = preprocess(data_test)
x_test = scaled_test.drop(['label'], axis=1).values
y_test = scaled_test['label'].values
y_test = y_test.astype('int')

pca_test = PCA(n_components=20)
x_test_reduced = pca_test.fit_transform(x_test)
y_test_reduced = y_test




printlog(f"[{get_ts()}] Training set original features: {x_train.shape[1]}, reduced features: {x_train_reduced.shape[1]}")
printlog(f"[{get_ts()}] Testing set original features: {x_test.shape[1]}, reduced features: {x_test_reduced.shape[1]}")


kernal_evals = dict()
def evaluate_classification(model, name, X_train, X_test, y_train, y_test):
    printlog(f"[{get_ts()}] Evaluating classifier: {name}...")
    start_time = time.time()
    train_accuracy = metrics.accuracy_score(y_train, model.predict(X_train))
    test_accuracy = metrics.accuracy_score(y_test, model.predict(X_test))
    
    train_precision = metrics.precision_score(y_train, model.predict(X_train))
    test_precision = metrics.precision_score(y_test, model.predict(X_test))
    
    train_recall = metrics.recall_score(y_train, model.predict(X_train))
    test_recall = metrics.recall_score(y_test, model.predict(X_test))
    
    kernal_evals[str(name)] = [train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall]
    
    end_time = time.time()
    printlog(f"[{get_ts()}] Testing time: {(end_time - start_time):.4f}")

    row1 = f"[{get_ts()}] Generating results...\n"
    row2 = f"[{get_ts()}] " + "Training Accuracy " + str(name) + " {}  Test Accuracy ".format(train_accuracy*100) + str(name) + " {}".format(test_accuracy*100) + "\n"
    row3 = f"[{get_ts()}] " + "Training Precesion " + str(name) + " {}  Test Precesion ".format(train_precision*100) + str(name) + " {}".format(test_precision*100) + "\n"
    row4 = f"[{get_ts()}] " + "Training Recall " + str(name) + " {}  Test Recall ".format(train_recall*100) + str(name) + " {}".format(test_recall*100)

    printlog(row1 + row2 + row3 + row4)
    
    actual = y_test
    predicted = model.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=['Normal', 'Attack'])
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.grid(False)
    cm_display.plot(ax=ax)
    counter = get_filename_counter()
    plt.savefig(os.path.join(output_dir, f"{counter}{name}_confusion_matrix.png"))
    print(f"[{get_ts()}] Saved results to {output_dir}/{counter}{name}_confusion_matrix.png", flush=True)

def f_importances(coef, names, top=-1, title="untitled"):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))

    # Show all features
    if top == -1:
        top = len(names)
    
    plt.figure(figsize=(10, 10))
    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.title(f'Feature importances for {title}')
    counter = get_filename_counter()
    plt.savefig(os.path.join(output_dir, f"{counter}Feature importances for {title}.png"))
    print(f"[{get_ts()}] Saved results to {output_dir}/{counter}Feature importances for {title}.png", flush=True)
    if (display_results): plt.show()

if (bool_lr): 
    printlog(f"[{get_ts()}] Preparing Logistic Regression")
    start_time = time.time()
    lr = LogisticRegression(**lr_params).fit(x_train, y_train)
    end_time = time.time()
    printlog(f"[{get_ts()}] Training time: {(end_time - start_time):.4f}")
    evaluate_classification(lr, "Logistic Regression", x_train, x_test, y_train, y_test)
if (bool_knn): 
    printlog(f"[{get_ts()}] Preparing KNeighborsClassifier")
    start_time = time.time()
    knn = KNeighborsClassifier(**knn_params).fit(x_train, y_train)
    end_time = time.time()
    printlog(f"[{get_ts()}] Training time: {(end_time - start_time):.4f}")
    evaluate_classification(knn, "KNeighborsClassifier", x_train, x_test, y_train, y_test)
if (bool_gnb): 
    printlog(f"[{get_ts()}] Preparing GaussianNB")
    start_time = time.time()
    gnb = GaussianNB(**gnb_params).fit(x_train, y_train)
    end_time = time.time()
    printlog(f"[{get_ts()}] Training time: {(end_time - start_time):.4f}")
    evaluate_classification(gnb, "GaussianNB", x_train, x_test, y_train, y_test)
if (bool_lin_svc): 
    printlog(f"[{get_ts()}] Preparing Linear SVC(LBasedImpl)")
    start_time = time.time()
    lin_svc = svm.LinearSVC(**lin_svc_params).fit(x_train, y_train)
    end_time = time.time()
    printlog(f"[{get_ts()}] Training time: {(end_time - start_time):.4f}")
    evaluate_classification(lin_svc, "Linear SVC(LBasedImpl)", x_train, x_test, y_train, y_test)
if (bool_dt): 
    printlog(f"[{get_ts()}] Preparing Decision Tree")
    start_time = time.time()
    dt = DecisionTreeClassifier(**dt_params).fit(x_train, y_train)
    tdt = DecisionTreeClassifier(**dt_params).fit(x_train, y_train)
    end_time = time.time()
    printlog(f"[{get_ts()}] Training time: {(end_time - start_time):.4f}")
    evaluate_classification(tdt, "DecisionTreeClassifier", x_train, x_test, y_train, y_test)


    features_names = data_train.drop(['label'], axis=1)
    f_importances(abs(tdt.feature_importances_), features_names, top=18, title="Decision Tree")

    print(f"[{get_ts()}] Generating results...", flush=True)
    fig = plt.figure(figsize=(60, 40))
    tree.plot_tree(dt, filled=True, feature_names=features_names.columns, fontsize=8)
    counter = get_filename_counter()
    plt.savefig(os.path.join(output_dir, f"{counter}Decision_tree.png"))
    print(f"[{get_ts()}] Saved results to {output_dir}/{counter}Decision_tree.png", flush=True)

if (bool_rf): 
    printlog(f"[{get_ts()}] Preparing RandomForest")
    start_time = time.time()
    rf = RandomForestClassifier(**rf_params).fit(x_train, y_train)
    end_time = time.time()
    printlog(f"[{get_ts()}] Training time: {(end_time - start_time):.4f}")
    evaluate_classification(rf, "RandomForestClassifier", x_train, x_test, y_train, y_test)
    features_names = data_train.drop(['label'], axis=1)
    f_importances(abs(rf.feature_importances_), features_names, top=18, title="Random Forest")

    printlog(f"[{get_ts()}] Preparing Reduced RandomForest")
    start_time = time.time()
    rrf = RandomForestClassifier(**rf_params).fit(x_train_reduced, y_train_reduced)
    end_time = time.time()
    printlog(f"[{get_ts()}] Training time: {(end_time - start_time):.4f}")
    evaluate_classification(rrf, "Reduced RandomForest", x_train_reduced, x_test_reduced, y_train_reduced, y_test_reduced)

if (bool_lr): 
    printlog(f"[{get_ts()}] Preparing XGBoost")
    start_time = time.time()
    xg_r = xgb.XGBRegressor(**xgb_params).fit(x_train_reduced, y_train_reduced)
    end_time = time.time()
    printlog(f"[{get_ts()}] Training time: {(end_time - start_time):.4f}")
    name = "XGBOOST"
    train_error = metrics.mean_squared_error(y_train_reduced, xg_r.predict(x_train_reduced), squared=False)
    test_error = metrics.mean_squared_error(y_test_reduced, xg_r.predict(x_test_reduced), squared=False)
    printlog(f"[{get_ts()}] " + "Training Error " + str(name) + " {}  Test error ".format(train_error) + str(name) + " {}".format(test_error))

    y_pred = xg_r.predict(x_test_reduced)
    df = pd.DataFrame({"Y_test": y_test_reduced, "Y_pred": y_pred})
    plt.figure(figsize=(16, 8))
    plt.plot(df[:80])
    plt.legend(['Actual', 'Predicted'])
    # if (display_results): plt.show()

log.close()
print(f"[{get_ts()}] End of program")
