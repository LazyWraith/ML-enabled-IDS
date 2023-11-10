import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
# from tensorflow.keras import regularizers
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
pd.set_option('display.max_columns',None)
warnings.filterwarnings('ignore')

###--------SETTINGS--------###
#
display_results = False
#
##############################

# data output
output_dir = "./output/NSL-KDD"
log = open(f'{output_dir}/log.txt', 'w')
filename_counter = 0

def printlog(message):
    log.write(message + "\n")
    print(message, flush=True)

def get_filename_counter():
    global filename_counter
    filename_counter += 1
    return str(filename_counter) + ". "

import time
start_ts = time.time()

def get_ts():
    ts = time.time() - start_ts
    return f"{ts:.4f}"


printlog(f"[{get_ts()}] Init complete!")

train_path = "./input/nsl-kdd/KDDTrain+.txt"
test_path = "./input/nsl-kdd/KDDTest+.txt"
# Read Train and Test dataset
printlog(f"[{get_ts()}] Reading from {train_path}")
data_train = pd.read_csv(train_path)
# data_train = pd.read_csv("./input/UNSW-NB15/UNSW_NB15_training-set.csv")
data_train.head()

columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','outcome','level'])

# columns = (['id','dur','proto','service','state','spkts','dpkts','sbytes','dbytes','rate','sttl','dttl','sload','dload','sloss','dloss','sinpkt','dinpkt','sjit','djit','swin','stcpb','dtcpb','dwin','tcprtt','synack','ackdat','smean','dmean','trans_depth','response_body_len','ct_srv_src','ct_state_ttl','ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm','ct_srv_dst','is_sm_ips_ports','attack_cat','label'])

# Assign names for columns
data_train.columns = columns

data_train.info()

data_train.describe().style.background_gradient(cmap='Blues').set_properties(**{'font-family':'Segoe UI'})

printlog(f"[{get_ts()}] Mapping outcomes...")
data_train.loc[data_train['outcome'] == "normal", "outcome"] = 'normal'
data_train.loc[data_train['outcome'] != 'normal', "outcome"] = 'attack'

def printlog(message):
    log.write(message + "\n")
    print(message, flush=True)

def pie_plot(df, cols_list, rows, cols):
    print(f"[{get_ts()}] Generating results...", flush=True)
    fig, axes = plt.subplots(rows, cols)
    for ax, col in zip(axes.ravel(), cols_list):
        df[col].value_counts().plot(ax=ax, kind='pie', figsize=(15, 15), fontsize=10, autopct='%1.0f%%')
        ax.set_title(str(col), fontsize = 12)
    counter = get_filename_counter()
    plt.savefig(os.path.join(output_dir, f"{counter}{col}_pie_chart.png"))  # Save the chart
    print(f"[{get_ts()}] Saved results to {output_dir}/{counter}{col}_pie_chart.png", flush=True)
    if (display_results): plt.show()

def Scaling(df_num, cols):
    std_scaler = RobustScaler()
    std_scaler_temp = std_scaler.fit_transform(df_num)
    std_df = pd.DataFrame(std_scaler_temp, columns =cols)
    return std_df

def preprocess(dataframe):
    print(f"[{get_ts()}] Running preprocess...", flush=True)
    df_num = dataframe.drop(cat_cols, axis=1)
    num_cols = df_num.columns
    scaled_df = Scaling(df_num, num_cols)
    
    dataframe.drop(labels=num_cols, axis="columns", inplace=True)
    dataframe[num_cols] = scaled_df[num_cols]
    
    dataframe.loc[dataframe['outcome'] == "normal", "outcome"] = 0
    dataframe.loc[dataframe['outcome'] != 0, "outcome"] = 1

    dataframe = pd.get_dummies(dataframe, columns = ['protocol_type', 'service', 'flag'])
    return dataframe


pie_plot(data_train, ['protocol_type', 'outcome'], 1, 2)

cat_cols = ['is_host_login','protocol_type','service','flag','land', 'logged_in','is_guest_login', 'level', 'outcome']


scaled_train = preprocess(data_train)

x = scaled_train.drop(['outcome', 'level'] , axis = 1).values
y = scaled_train['outcome'].values
y_reg = scaled_train['level'].values

print(f"[{get_ts()}] Running PCA...", flush=True)
pca = PCA(n_components=20)
pca = pca.fit(x)
x_reduced = pca.transform(x)
printlog(f"[{get_ts()}] Number of original features is {x.shape[1]} and of reduced features is {x_reduced.shape[1]}")

y = y.astype('int')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train_reduced, x_test_reduced, y_train_reduced, y_test_reduced = train_test_split(x_reduced, y, test_size=0.2, random_state=42)
x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(x, y_reg, test_size=0.2, random_state=42)

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
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['normal', 'attack'])

    fig, ax = plt.subplots(figsize=(10,10))
    ax.grid(False)
    cm_display.plot(ax=ax)
    counter = get_filename_counter()
    plt.savefig(os.path.join(output_dir, f"{counter}{name}_confusion_matrix.png"))
    print(f"[{get_ts()}] Saved results to {output_dir}/{counter}{name}_confusion_matrix.png", flush=True)

lr = LogisticRegression().fit(x_train, y_train)
evaluate_classification(lr, "Logistic Regression", x_train, x_test, y_train, y_test)

knn = KNeighborsClassifier(n_neighbors=20).fit(x_train, y_train)
evaluate_classification(knn, "KNeighborsClassifier", x_train, x_test, y_train, y_test)

gnb = GaussianNB().fit(x_train, y_train)
evaluate_classification(gnb, "GaussianNB", x_train, x_test, y_train, y_test)

lin_svc = svm.LinearSVC().fit(x_train, y_train)
evaluate_classification(lin_svc, "Linear SVC(LBasedImpl)", x_train, x_test, y_train, y_test)

dt = DecisionTreeClassifier(max_depth=3).fit(x_train, y_train)
tdt = DecisionTreeClassifier().fit(x_train, y_train)
evaluate_classification(tdt, "DecisionTreeClassifier", x_train, x_test, y_train, y_test)

def f_importances(coef, names, top=-1, title="untitled"):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))

    # Show all features
    if top == -1:
        top = len(names)
    
    plt.figure(figsize=(10,10))
    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.title(f'Feature importances for {title}')
    counter = get_filename_counter()
    plt.savefig(os.path.join(output_dir, f"{counter}Feature importances for {title}.png"))
    print(f"[{get_ts()}] Saved results to {output_dir}/{counter}Feature importances for {title}.png", flush=True)
    if (display_results): plt.show()

features_names = data_train.drop(['outcome', 'level'] , axis = 1)
f_importances(abs(tdt.feature_importances_), features_names, top=18, title="Decision Tree")

fig = plt.figure(figsize=(15,12))
tree.plot_tree(dt , filled=True)
counter = get_filename_counter()
plt.savefig(os.path.join(output_dir, f"{counter}Decision_tree.png"))
print(f"[{get_ts()}] Saved results to {output_dir}/{counter}Decision_tree.png", flush=True)

rf = RandomForestClassifier().fit(x_train, y_train)
evaluate_classification(rf, "RandomForestClassifier", x_train, x_test, y_train, y_test)

f_importances(abs(rf.feature_importances_), features_names, top=18, title="Random Forest")

xg_r = xgb.XGBRegressor(objective ='reg:linear',n_estimators = 20).fit(x_train_reg, y_train_reg)
name = "XGBOOST"
train_error = metrics.mean_squared_error(y_train_reg, xg_r.predict(x_train_reg), squared=False)
test_error = metrics.mean_squared_error(y_test_reg, xg_r.predict(x_test_reg), squared=False)
printlog(f"[{get_ts()}] " + "Training Error " + str(name) + " {}  Test error ".format(train_error) + str(name) + " {}".format(test_error))

y_pred = xg_r.predict(x_test_reg)
df = pd.DataFrame({"Y_test": y_test_reg , "Y_pred" : y_pred})
plt.figure(figsize=(16,8))
plt.plot(df[:80])
plt.legend(['Actual' , 'Predicted'])

rrf = RandomForestClassifier().fit(x_train_reduced, y_train_reduced)
evaluate_classification(rrf, "PCA RandomForest", x_train_reduced, x_test_reduced, y_train_reduced, y_test_reduced)

log.close()
print(f"[{get_ts()}] " + "End of program")