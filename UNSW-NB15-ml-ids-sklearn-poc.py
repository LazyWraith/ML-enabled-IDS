import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import xgboost as xgb
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

###----ML-PARAMETERS-------###

# Logical Regression
lr_params = {
    "C": 4.970645224458411,
    "max_iter": 998,
    "solver": "liblinear"
}

# K-Nearest Neighbors
knn_params = {
    "n_neighbors": 3
}

# GaussianNB
gnb_params = {
    "var_smoothing": 1.2142824485311176e-12
}

# LinearSVC
lin_svc_params = {
    "C": 0.036763506533393595,
    "max_iter": 784
}

# Decision Trees
dt_params = {
    "max_depth": 15,
    "min_samples_split": 9,
    "min_samples_leaf": 1
}

# XGBoost
xgb_params = {
    "n_estimators": 129,
    "max_depth": 8,
    "learning_rate": 0.09579543793645304
}

# RandomForestClassifier
rf_params = {
    "n_estimators": 53,
    "max_depth": 15,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": 0.4414469827896669
}

##############################

# result output
output_dir = "./output/UNSW-NB15"
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
    return f"{ts:.4f}"

printlog(f"[{get_ts()}] Init complete!")

train_path = "./input/UNSW_NB15/UNSW_NB15_training-set.csv"
printlog(f"[{get_ts()}] Reading from {train_path}")

# Read Train and Test dataset
data_train = pd.read_csv(train_path)
data_train.head()
columns = (['id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'attack_cat', 'label'])

# Assign names for columns
data_train.columns = columns
data_train.info()
data_train.describe().style.background_gradient(cmap='Blues').set_properties(**{'font-family': 'Segoe UI'})

printlog(f"[{get_ts()}] Mapping outcomes...")
data_train.loc[data_train['label'] == "Normal", "label"] = 0
data_train.loc[data_train['label'] != 0, "label"] = 1

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

# def preprocess(dataframe):
#     df_num = dataframe.drop(cat_cols, axis=1)  # Drop 'attack_cat' and 'Label'
#     num_cols = df_num.columns
#     scaled_df = Scaling(df_num, num_cols)
    
#     dataframe.drop(labels=num_cols, axis="columns", inplace=True)
#     dataframe[num_cols] = scaled_df[num_cols]
    
#     dataframe = pd.get_dummies(dataframe, columns=['proto', 'service', 'state'])
#     return dataframe

def preprocess(dataframe):
    printlog(f"[{get_ts()}] Running preprocess...")
    # Drop 'attack_cat' and 'label'
    df_num = dataframe.drop(cat_cols, axis=1)  
    
    # Separate numerical and categorical columns
    num_cols = df_num.select_dtypes(include=[np.number]).columns
    obj_cols = ['proto', 'service', 'state']

    # One-hot encode categorical columns
    dataframe = pd.get_dummies(dataframe, columns=obj_cols)

    # Separate numerical and label columns
    df_num = dataframe[num_cols]
    labels = dataframe['label']

    # Apply RobustScaler to numerical columns
    std_scaler = RobustScaler()
    std_scaler_temp = std_scaler.fit_transform(df_num)
    std_df = pd.DataFrame(std_scaler_temp, columns=num_cols)

    # Combine scaled numerical columns and labels
    dataframe = pd.concat([std_df, labels], axis=1)

    return dataframe

cat_cols = ['attack_cat', 'label']

pie_plot(data_train, ['proto', 'service'], 1, 2)

pie_plot(data_train, ['attack_cat', 'label'], 1, 2)

scaled_train = preprocess(data_train)

x = scaled_train.drop(['label'] , axis = 1).values
y = scaled_train['label'].values

pca = PCA(n_components=20)
pca = pca.fit(x)
x_reduced = pca.transform(x)
printlog(f"[{get_ts()}] Number of original features is {x.shape[1]} and of reduced features is {x_reduced.shape[1]}")

y = y.astype('int')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train_reduced, x_test_reduced, y_train_reduced, y_test_reduced = train_test_split(x_reduced, y, test_size=0.2, random_state=42)

kernal_evals = dict()
def evaluate_classification(model, name, X_train, X_test, y_train, y_test):
    printlog(f"[{get_ts()}] Evaluating classifier: {name}...")
    train_accuracy = metrics.accuracy_score(y_train, model.predict(X_train))
    test_accuracy = metrics.accuracy_score(y_test, model.predict(X_test))
    
    train_precision = metrics.precision_score(y_train, model.predict(X_train))
    test_precision = metrics.precision_score(y_test, model.predict(X_test))
    
    train_recall = metrics.recall_score(y_train, model.predict(X_train))
    test_recall = metrics.recall_score(y_test, model.predict(X_test))
    
    kernal_evals[str(name)] = [train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall]
    
    row1 = f"[{get_ts()}] Generating results...\n"
    row2 = f"[{get_ts()}] " + "Training Accuracy " + str(name) + " {}  Test Accuracy ".format(train_accuracy*100) + str(name) + " {}".format(test_accuracy*100) + "\n"
    row3 = f"[{get_ts()}] " + "Training Precesion " + str(name) + " {}  Test Precesion ".format(train_precision*100) + str(name) + " {}".format(test_precision*100) + "\n"
    row4 = f"[{get_ts()}] " + "Training Recall " + str(name) + " {}  Test Recall ".format(train_recall*100) + str(name) + " {}".format(test_recall*100)

    printlog(row1 + row2 + row3 + row4)
    
    actual = y_test
    predicted = model.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=['Normal', 'Attack'])
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.grid(False)
    cm_display.plot(ax=ax)
    counter = get_filename_counter()
    plt.savefig(os.path.join(output_dir, f"{counter}{name}_confusion_matrix.png"))
    print(f"[{get_ts()}] Saved results to {output_dir}/{counter}{name}_confusion_matrix.png", flush=True)


lr = LogisticRegression(**lr_params).fit(x_train, y_train)
evaluate_classification(lr, "Logistic Regression", x_train, x_test, y_train, y_test)

knn = KNeighborsClassifier(**knn_params).fit(x_train, y_train)
evaluate_classification(knn, "KNeighborsClassifier", x_train, x_test, y_train, y_test)

gnb = GaussianNB(**gnb_params).fit(x_train, y_train)
evaluate_classification(gnb, "GaussianNB", x_train, x_test, y_train, y_test)

lin_svc = svm.LinearSVC(**lin_svc_params).fit(x_train, y_train)
evaluate_classification(lin_svc, "Linear SVC(LBasedImpl)", x_train, x_test, y_train, y_test)

dt = DecisionTreeClassifier(**dt_params).fit(x_train, y_train)
tdt = DecisionTreeClassifier(**dt_params).fit(x_train, y_train)
evaluate_classification(tdt, "DecisionTreeClassifier", x_train, x_test, y_train, y_test)

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

features_names = data_train.drop(['label'], axis=1)
f_importances(abs(tdt.feature_importances_), features_names, top=18, title="Decision Tree")

print(f"[{get_ts()}] Generating results...", flush=True)
fig = plt.figure(figsize=(45, 32))
tree.plot_tree(dt, filled=True)
counter = get_filename_counter()
plt.savefig(os.path.join(output_dir, f"{counter}Decision_tree.png"))
print(f"[{get_ts()}] Saved results to {output_dir}/{counter}Decision_tree.png", flush=True)


rf = RandomForestClassifier(**rf_params).fit(x_train, y_train)
evaluate_classification(rf, "RandomForestClassifier", x_train, x_test, y_train, y_test)

f_importances(abs(rf.feature_importances_), features_names, top=18, title="Random Forest")

xg_r = xgb.XGBRegressor(**xgb_params).fit(x_train_reduced, y_train_reduced)
name = "XGBOOST"
train_error = metrics.mean_squared_error(y_train_reduced, xg_r.predict(x_train_reduced), squared=False)
test_error = metrics.mean_squared_error(y_test_reduced, xg_r.predict(x_test_reduced), squared=False)
printlog(f"[{get_ts()}] " + "Training Error " + str(name) + " {}  Test error ".format(train_error) + str(name) + " {}".format(test_error))

y_pred = xg_r.predict(x_test_reduced)
df = pd.DataFrame({"Y_test": y_test_reduced, "Y_pred": y_pred})
plt.figure(figsize=(16, 8))
plt.plot(df[:80])
plt.legend(['Actual', 'Predicted'])

rrf = RandomForestClassifier(**rf_params).fit(x_train_reduced, y_train_reduced)
evaluate_classification(rrf, "PCA RandomForest", x_train_reduced, x_test_reduced, y_train_reduced, y_test_reduced)

log.close()
print(f"[{get_ts()}] End of program")
