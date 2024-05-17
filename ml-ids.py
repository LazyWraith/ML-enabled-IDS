import csv
from collections import Counter
import os
import traceback
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
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn import metrics
from imblearn.over_sampling import SMOTE
import pickle
import json
import time
start_ts = time.time()
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

with open('settings.json', 'r') as json_file:
    settings = json.load(json_file)

display_results = settings.get('display_results')
generate_statistics_pie = settings.get('generate_statistics_pie', True)
dataset_name = settings.get('dataset_name', 'UNSW-NB15')
output_dir = settings.get('output_dir')
load_saved_models = settings.get('load_saved_models', True)
save_trained_models = not load_saved_models
model_save_path = settings.get('model_save_path', './Saved models')
model_save_version = settings.get('model_save_version')
model_save_path = f"{model_save_path}/{model_save_version}"
eval_average = settings.get('average')
use_multiclass = settings.get('multiclass')

bool_gnb = settings.get('bool_gnb', True)
bool_xgb = settings.get('bool_xgb', True)
bool_et = settings.get('bool_et', True)
bool_dt = settings.get('bool_dt', True)
bool_rf = settings.get('bool_rf', True)
bool_lr = settings.get('bool_lr', True)
bool_lin_svc = settings.get('bool_lin_svc', True)
bool_knn = settings.get('bool_knn', True)
bool_dnn = settings.get('bool_dnn', True)

use_single_dataset = settings.get('use_single_dataset', True)
split_train_ratio = settings.get('split_train_ratio', 0.6)
split_test_ratio = 1 - split_train_ratio
rndm_state = settings.get('rndm_state', 42)

filename_counter = 0

def get_filename_counter():
    global filename_counter
    filename_counter += 1
    return str(filename_counter) + ". "

def printlog(message):
    message = str(message)
    log.write(message + "\n")
    print(message, flush=True)

def get_ts():
    ts = time.time() - start_ts
    return f"{ts:>08.4f}"

def save_model(model, name):
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(model_save_path, f"{model_save_version} {name}.pkl")
    printlog(f"[{get_ts()}] Saving model to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(name):
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(model_save_path, f"{model_save_version} {name}.pkl")
    printlog(f"[{get_ts()}] Loading model from {filename}")
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def pie_plot(df, cols_list, rows, cols):
    fig, axes = plt.subplots(rows, cols)
    for ax, col in zip(axes.ravel(), cols_list):
        df[col].value_counts().plot(ax=ax, kind='pie', figsize=(12, 8), fontsize=10, autopct='%1.0f%%')
        ax.set_title(str(col), fontsize=12)
    counter = get_filename_counter()
    plt.savefig(os.path.join(output_dir, f"{counter}{col}_pie_chart.png"))  # Save the chart
    print(f"[{get_ts()}] Saved results to {output_dir}/{counter}{col}_pie_chart.png", flush=True)
    if (display_results): plt.show()
    plt.clf()

def cm_plot(y_test, y_predict, name):
    # Combine all non-"Normal" classes into a single "Attack" class
        # Flatten CM attack classes into one
    y_test_combined = np.where(y_test == 0, 0, 1)
    test_predict_combined = np.where(y_predict == 0, 0, 1)
    cm = metrics.confusion_matrix(y_test_combined, test_predict_combined)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Attack'])
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.grid(False)
    cm_display.plot(ax=ax)
    counter = get_filename_counter()
    plt.savefig(os.path.join(output_dir, f"{counter}{name}_confusion_matrix.png"))
    print(f"[{get_ts()}] Saved results to {output_dir}/{counter}{name}_confusion_matrix.png", flush=True)
    plt.clf()
    
    if use_multiclass:
        try:
            cm=metrics.confusion_matrix(y_test, y_predict)
            f,ax=plt.subplots(figsize=(5,5))
            sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
            plt.xlabel("y_pred")
            plt.ylabel("y_true")
            # plt.show()
            plt.savefig(os.path.join(output_dir, f"{counter}{name}_multi_confusion_matrix.png"))
            print(f"[{get_ts()}] Saved results to {output_dir}/{counter}{name}_multi_confusion_matrix.png", flush=True)
            plt.clf()
        except Exception as e:
            print(e)
            print(f"Unable to plot CM for {name}")

def roc_plot(fpr, tpr, label1 = 'ROC Curve', label2='Random guess', title=''):
    # Plot ROC curve
    plt.plot(fpr, tpr, label=label1)
    plt.plot([0, 1], [0, 1], 'k--', label=label2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title + 'ROC Curve')
    plt.legend()
    counter = get_filename_counter()
    plt.savefig(os.path.join(output_dir, f"{counter}{title}_roc.png"))
    print(f"[{get_ts()}] Saved results to {output_dir}/{counter}{title}_roc.png", flush=True)
    plt.clf()

def preprocess(dataframe):
    dataframe = dataframe.drop(drop_cols, axis=1)
    df_num = dataframe.drop(cat_cols, axis=1)
    num_cols = df_num.select_dtypes(include=[np.number]).columns
    dataframe = pd.get_dummies(dataframe, columns=obj_cols)
    df_num = dataframe[num_cols]
    labels = dataframe[label_header]
    
    # Replace NaN values with 0
    df_num.fillna(0, inplace=True)
    
    # Replace infinity values with 0
    df_num.replace([np.inf, -np.inf], 0, inplace=True)
    
    # std_scaler = RobustScaler()
    # std_scaler_temp = std_scaler.fit_transform(df_num)
    # std_df = pd.DataFrame(std_scaler_temp, columns=num_cols)
    # dataframe = pd.concat([std_df, labels], axis=1)
    dataframe = pd.concat([df_num, labels], axis=1)
    return dataframe

def save_metrics_to_csv(eval_metrics, filepath=f"{output_dir}/results.csv"):
    """
    Save evaluation metrics to a CSV file.

    Parameters:
    - metrics_dict: A dictionary containing evaluation metrics.
    - csv_file_path: Path to the CSV file (default is 'evaluation_metrics.csv').
    """
    with open(filepath, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)

        # Write the header row
        header = ['Model', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1 Score']
        csv_writer.writerow(header)

        # Write metrics for each model
        for model_name, metrics_list in eval_metrics.items():
            row_data = [model_name] + metrics_list
            csv_writer.writerow(row_data)

    print(f"Metrics saved to {filepath}")


def smote_balancing(x_train, y_train, jobs):
    printlog("Performing SMOTE balancing...")
    for data in jobs:
        smote=SMOTE(n_jobs=-1,sampling_strategy={data[0]:data[1]})
        x_train, y_train = smote.fit_resample(x_train, y_train)
    result = pd.Series(y_train).value_counts()
    for encoded_value, count in result.items():
        original_label = label_mapping[encoded_value].replace('�', '-')
        printlog(f"{original_label} ({encoded_value}): {count}")

    return x_train, y_train

def evaluate_classification(model, name, x_train, x_test, y_train, y_test):
    printlog(f"[{get_ts()}] Evaluating classifier: {name}...")
    start_time = time.time()
    test_predict = model.predict(x_test)
    test_accuracy = metrics.accuracy_score(y_test, test_predict)
    test_precision = metrics.precision_score(y_test, test_predict, average=eval_average)
    test_recall = metrics.recall_score(y_test, test_predict, average=eval_average)
    test_f1 = metrics.f1_score(y_test, test_predict, average=eval_average)

    report = metrics.classification_report(y_test, test_predict)
    printlog(report)
    kernal_evals[str(name)] = [test_accuracy, test_precision, test_recall, test_f1]
    
    end_time = time.time()
    printlog(f"[{get_ts()}] Testing time: {(end_time - start_time):.4f}")

    cm_plot(y_test, test_predict, name)

    try:
        test_predict_proba = model.predict_proba(x_test)[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba)
        roc_auc = metrics.auc(fpr, tpr)
        roc_plot(fpr, tpr, label1=f"AUC: {roc_auc}", title=name)
    except Exception as e:
        printlog(f"[{get_ts()}] Failed to create ROC for: {name}!")
        # traceback.print_exc()
    return test_predict

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
    plt.clf()

def run_lr(x_train, y_train, x_test, y_test):
    file_name = "Logistic Regression"
    if load_saved_models:
        lr = load_model(file_name)
    else:
        printlog(f"[{get_ts()}] Preparing Logistic Regression")
        start_time = time.time()
        lr = LogisticRegression(**lr_params).fit(x_train, y_train)
        end_time = time.time()
        printlog(f"[{get_ts()}] Training time: {(end_time - start_time):.4f}")
    evaluate_classification(lr, "Logistic Regression", x_train, x_test, y_train, y_test)
    if save_trained_models: save_model(lr, file_name)

def run_knn(x_train, y_train, x_test, y_test):
    file_name = "KNeighborsClassifier"
    if load_saved_models:
        knn = load_model(file_name)
    else:
        printlog(f"[{get_ts()}] Preparing KNeighborsClassifier")
        start_time = time.time()
        knn = KNeighborsClassifier(**knn_params).fit(x_train, y_train)
        end_time = time.time()
        printlog(f"[{get_ts()}] Training time: {(end_time - start_time):.4f}")
    evaluate_classification(knn, "KNeighborsClassifier", x_train, x_test, y_train, y_test)
    if save_trained_models: save_model(knn, file_name)

def run_gnb(x_train, y_train, x_test, y_test):
    file_name = "GaussianNB"
    if load_saved_models:
        gnb = load_model(file_name)
    else:
        printlog(f"[{get_ts()}] Preparing GaussianNB")
        start_time = time.time()
        gnb = GaussianNB(**gnb_params).fit(x_train, y_train)
        end_time = time.time()
        printlog(f"[{get_ts()}] Training time: {(end_time - start_time):.4f}")
    evaluate_classification(gnb, "GaussianNB", x_train, x_test, y_train, y_test)
    if save_trained_models: save_model(gnb, file_name)

def run_lin_svc(x_train, y_train, x_test, y_test): 
    file_name = "Linear SVC(LBasedImpl)"
    if load_saved_models:
        lin_svc = load_model(file_name)
    else:
        printlog(f"[{get_ts()}] Preparing Linear SVC(LBasedImpl)")
        start_time = time.time()
        lin_svc = svm.LinearSVC(**lin_svc_params).fit(x_train, y_train)
        end_time = time.time()
        printlog(f"[{get_ts()}] Training time: {(end_time - start_time):.4f}")
    evaluate_classification(lin_svc, "Linear SVC(LBasedImpl)", x_train, x_test, y_train, y_test)
    if save_trained_models: save_model(lin_svc, file_name)

def run_dt(x_train, y_train, x_test, y_test):
    file_name = "DecisionTreeClassifier"
    if load_saved_models:
        dt = load_model(file_name)
        tdt = load_model(file_name)
    else:
        printlog(f"[{get_ts()}] Preparing Decision Tree")
        start_time = time.time()
        dt = DecisionTreeClassifier(**dt_params).fit(x_train, y_train)
        tdt = DecisionTreeClassifier(**dt_params).fit(x_train, y_train)
        end_time = time.time()
        printlog(f"[{get_ts()}] Training time: {(end_time - start_time):.4f}")
    evaluate_classification(tdt, "DecisionTreeClassifier", x_train, x_test, y_train, y_test)
    if save_trained_models: save_model(tdt, file_name)
    features_names = data_train.drop(label_header, axis=1)
    f_importances(abs(tdt.feature_importances_), features_names, top=18, title="Decision Tree")

    print(f"[{get_ts()}] Generating results...", flush=True)
    fig = plt.figure(figsize=(60, 40))
    tree.plot_tree(dt, filled=True, feature_names=features_names.columns, fontsize=8)
    counter = get_filename_counter()
    plt.savefig(os.path.join(output_dir, f"{counter}Decision_tree.png"))
    print(f"[{get_ts()}] Saved results to {output_dir}/{counter}Decision_tree.png", flush=True)
    plt.clf()

def run_rf(x_train, y_train, x_test, y_test):
    file_name = "RandomForestClassifier"
    if load_saved_models:
        rf = load_model(file_name)
    else:
        printlog(f"[{get_ts()}] Preparing RandomForest")
        start_time = time.time()
        rf = RandomForestClassifier(**rf_params).fit(x_train, y_train)
        end_time = time.time()
        printlog(f"[{get_ts()}] Training time: {(end_time - start_time):.4f}")
    evaluate_classification(rf, "RandomForestClassifier", x_train, x_test, y_train, y_test)
    features_names = data_train.drop(label_header, axis=1)
    f_importances(abs(rf.feature_importances_), features_names, top=18, title="Random Forest")
    if save_trained_models: save_model(rf, file_name)

def run_xgb(x_train, y_train, x_test, y_test):
    file_name = "XGBoost"
    if load_saved_models:
        xg_c = load_model(file_name)
    else:
        printlog(f"[{get_ts()}] Preparing XGBoost")
        start_time = time.time()
        xg_c = xgb.XGBClassifier(**xgb_params).fit(x_train, y_train)
        end_time = time.time()
        printlog(f"[{get_ts()}] Training time: {(end_time - start_time):.4f}")
    evaluate_classification(xg_c, file_name, x_train, x_test, y_train, y_test)
    if save_trained_models: save_model(xg_c, file_name)

def run_et(x_train, y_train, x_test, y_test):
    file_name = "ExtraTrees"
    if load_saved_models:
        et = load_model(file_name)
    else:
        printlog(f"[{get_ts()}] Preparing ExtraTrees")
        start_time = time.time()
        et = ExtraTreesClassifier(**et_params).fit(x_train, y_train)
        end_time = time.time()
        printlog(f"[{get_ts()}] Training time: {(end_time - start_time):.4f}")
    evaluate_classification(et, file_name, x_train, x_test, y_train, y_test)
    if save_trained_models: save_model(et, file_name)

def run_dnn(x_train, y_train, x_test, y_test):
    file_name = "Deep Neural Network"
    if load_saved_models:
        dnn = load_model(file_name)
    else:
        printlog(f"[{get_ts()}] Preparing DNN")
        start_time = time.time()
        dnn = tf.keras.Sequential()
        for units in dnn_params["dense_layers"]:
            dnn.add(tf.keras.layers.Dense(units, activation=dnn_params["activation"], input_shape=(x_train.shape[1],)))
            dnn.add(tf.keras.layers.Dropout(dnn_params["dropout_rate"]))
        # Output Dense layer
        dnn.add(tf.keras.layers.Dense(1, activation=dnn_params["output_activation"]))
        dnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=dnn_params["learning_rate"]), loss=dnn_params["loss"], metrics=dnn_params["metrics"])
        results = dnn.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))
        end_time = time.time()
        printlog(f"[{get_ts()}] Training time: {(end_time - start_time):.4f}")
        plt.subplots(figsize=(8, 6))
        plt.plot(results.history['accuracy'], label='Training Accuracy')
        plt.plot(results.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        counter = get_filename_counter()
        plt.savefig(os.path.join(output_dir, f"{counter}Deep Neural Network.png"))
        print(f"[{get_ts()}] Saved results to {output_dir}/{counter}Deep Neural Network.png", flush=True)
        if(display_results): plt.show()
        plt.clf()

    loss, accuracy = dnn.evaluate(x_test, y_test)
    printlog(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
    if save_trained_models: save_model(dnn, file_name)

    actual = y_test
    # predicted = dnn.predict(x_test)
    evaluate_classification(dnn, file_name, x_train, x_test, y_train, y_test)

def run_models(x_train, y_train, x_test, y_test):
    if (bool_gnb): run_gnb(x_train, y_train, x_test, y_test)
    if (bool_xgb): run_xgb(x_train, y_train, x_test, y_test)
    if (bool_et): run_et(x_train, y_train, x_test, y_test)
    if (bool_dt): run_dt(x_train, y_train, x_test, y_test)
    if (bool_rf): run_rf(x_train, y_train, x_test, y_test)
    if (bool_lr): run_lr(x_train, y_train, x_test, y_test)
    if (bool_lin_svc): run_lin_svc(x_train, y_train, x_test, y_test)
    if (bool_knn): run_knn(x_train, y_train, x_test, y_test)
    if (bool_dnn): run_dnn(x_train, y_train, x_test, y_test)

# def run_models_reduced(x_train_reduced, y_train_reduced, x_test_reduced, y_test_reduced):
#     if (bool_rf): run_rrf(x_train_reduced, y_train_reduced, x_test_reduced, y_test_reduced)
#     if (bool_xgb): run_xgb(x_train_reduced, y_train_reduced, x_test_reduced, y_test_reduced)
    



#-------------------------------- MAIN --------------------------------

# Read dataset configuration from JSON
with open('dataset-config.json', 'r') as file:
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
        columns = config.get("columns")
    cat_cols = config.get("cat_cols")
    obj_cols = config.get("obj_cols")
    drop_cols = config.get("drop_cols")
    label_header = config.get("label_header")
    label_normal_value = config.get("label_normal_value")
    pie_stats = config.get("pie_stats")
    feature_reduced_number = config.get('feature_reduced_number')
    resampling_job = config.get('resampling_job')
    
else:
    print("Invalid dataset name!")

###----ML-PARAMETERS-------###
with open('./Hyperparameter Tuning/hyperparameters.json', 'r') as file:
    hyperparameters = json.load(file)

lr_params = hyperparameters.get("lr_params", {})
knn_params = hyperparameters.get("knn_params", {})
gnb_params = hyperparameters.get("gnb_params", {})
lin_svc_params = hyperparameters.get("lin_svc_params", {})
dt_params = hyperparameters.get("dt_params", {})
et_params = hyperparameters.get("et_params", {})
xgb_params = hyperparameters.get("xgb_params", {})
rf_params = hyperparameters.get("rf_params", {})
dnn_params = hyperparameters.get("dnn_params", {})
##############################

Path(output_dir).mkdir(parents=True, exist_ok=True)
# result output
log = open(f'{output_dir}/log.txt', 'w')

printlog(f"[{get_ts()}] Init complete!")
printlog(f"[{get_ts()}] Reading from {train_path}")

# Read Train and Test dataset
data_train = pd.read_csv(train_path)
if not use_single_dataset: 
    data_test = pd.read_csv(test_path)
else: 
    data_test = data_train
    
if (read_cols_from_csv): 
    # columns = data_train.columns.tolist()
    # Removes white spaces
    columns = data_train.columns.str.strip().tolist()

# Assign names for columns
data_train.columns = columns
data_test.columns = columns
data_train.info()
if not use_single_dataset: data_test.info()
# data_train.describe().style.background_gradient(cmap='Blues').set_properties(**{'font-family': 'Segoe UI'})
kernal_evals = dict()

printlog(f"[{get_ts()}] Mapping outcomes...")

if generate_statistics_pie:
    for i in pie_stats:
        pie_plot(data_train, i, 1, 2)

# Process and split dataset
if use_single_dataset: 
    if use_multiclass:
        labelencoder = LabelEncoder()
        data_train.iloc[:, -1] = labelencoder.fit_transform(data_train.iloc[:, -1])
        label_mapping = {index: label for index, label in enumerate(labelencoder.classes_)}

    else:
        data_train.loc[data_train[label_header] == label_normal_value, label_header] = 0
        data_train.loc[data_train[label_header] != 0, label_header] = 1
    
    scaled_train = preprocess(data_train)

    x = scaled_train.drop(label_header , axis = 1).values
    y = scaled_train[label_header].values

    value_counts = data_train[label_header].value_counts()
    if use_multiclass:
        for encoded_value, count in value_counts.items():
            original_label = label_mapping[encoded_value].replace('�', '-')
            printlog(f"{original_label} ({encoded_value}): {count}")
    else:
        printlog(value_counts)
    y=np.ravel(y)
    y = y.astype('int')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_test_ratio, random_state=rndm_state)

    unique_attack_cats = np.unique(y_train)
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # SMOTE
    if use_multiclass:
        x_train, y_train = smote_balancing(x_train, y_train, resampling_job)
    run_models(x_train, y_train, x_test, y_test)

# Evaluate using separate train and test datasets
else:
    if use_multiclass:
        labelencoder = LabelEncoder()
        data_train.iloc[:, -1] = labelencoder.fit_transform(data_train.iloc[:, -1])
        data_test.iloc[:, -1] = labelencoder.fit_transform(data_test.iloc[:, -1])
        label_mapping = {index: label for index, label in enumerate(labelencoder.classes_)}

    else:
        # binary classification
        data_train.loc[data_train[label_header] == label_normal_value, label_header] = 0
        data_train.loc[data_train[label_header] != 0, label_header] = 1

        data_test.loc[data_test[label_header] == label_normal_value, label_header] = 0
        data_test.loc[data_test[label_header] != 0, label_header] = 1
    # Process training set
    
    scaled_train = preprocess(data_train)
    x_train = scaled_train.drop(label_header, axis=1).values
    y_train = scaled_train[label_header].values
    y_train = y_train.astype('int')

    pca_train = PCA(n_components=feature_reduced_number)
    x_train_reduced = pca_train.fit_transform(x_train)
    y_train_reduced = y_train

    # Process testing set
    scaled_test = preprocess(data_test)
    x_test = scaled_test.drop(label_header, axis=1).values
    y_test = scaled_test[label_header].values
    y_test = y_test.astype('int')

    run_models(x_train, y_train, x_test, y_test)

# Save all metrics to file
save_metrics_to_csv(kernal_evals)
log.close()
print(f"[{get_ts()}] End of program")
