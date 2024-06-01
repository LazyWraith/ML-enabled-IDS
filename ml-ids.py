import csv
import multiprocessing
from multiprocessing import Pool
import concurrent.futures
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
import logging
import logging.handlers

class Ml:
    def configure_logging():
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()  # Log to console
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Remove existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()
        
        logger.addHandler(handler)

    def __init__(self):
        self.start_ts = time.time()
        with open('settings.json', 'r') as json_file:
            settings = json.load(json_file)

        self.display_results = settings.get('display_results')
        self.generate_statistics_pie = settings.get('generate_statistics_pie', True)
        self.dataset_name = settings.get('dataset_name', 'UNSW-NB15')
        self.output_dir = settings.get('output_dir')
        self.load_saved_models = settings.get('load_saved_models', True)
        self.save_trained_models = settings.get('save_trained_models', False)
        self.model_save_path = settings.get('model_save_path', './Saved models')
        self.model_save_version = settings.get('model_save_version')
        self.model_save_path = f"{self.model_save_path}/{self.model_save_version}"
        self.eval_average = settings.get('average')
        self.max_workers = settings.get('max_workers')

        self.bool_gnb = settings.get('bool_gnb', True)
        self.bool_xgb = settings.get('bool_xgb', True)
        self.bool_et = settings.get('bool_et', True)
        self.bool_dt = settings.get('bool_dt', True)
        self.bool_rf = settings.get('bool_rf', True)
        self.bool_lr = settings.get('bool_lr', True)
        self.bool_lin_svc = settings.get('bool_lin_svc', True)
        self.bool_knn = settings.get('bool_knn', True)
        self.bool_dnn = settings.get('bool_dnn', True)

        self.use_single_dataset = settings.get('use_single_dataset', True)
        self.split_train_ratio = settings.get('split_train_ratio', 0.6)
        self.split_test_ratio = 1 - self.split_train_ratio
        self.rndm_state = settings.get('rndm_state', 42)

        self.filename_counter = 0
        self.kernal_evals = dict()
        self.eval_report = dict()
        
        # Read dataset configuration from JSON
        with open('dataset-config.json', 'r') as file:
            datasets_config = json.load(file)

        # Check if the dataset name is valid
        if self.dataset_name in datasets_config:
            config = datasets_config[self.dataset_name]

            # Dataset Path
            self.train_path = config["train_path"]
            self.test_path = config["test_path"]

            # Dataset Headers
            self.read_cols_from_csv = config.get("read_cols_from_csv", True)
            if (not self.read_cols_from_csv):
                self.columns = config.get("columns")

            self.cat_cols = config.get("cat_cols")
            self.obj_cols = config.get("obj_cols")
            self.drop_cols = config.get("drop_cols")
            self.label_header = config.get("label_header")
            self.label_normal_value = config.get("label_normal_value")
            self.pie_stats = config.get("pie_stats")
            self.feature_reduced_number = config.get('feature_reduced_number')
            self.resampling_job = config.get('resampling_job')
            self.class_mapping = config.get("class_mapping", {})
            self.num_classes = len(self.class_mapping)
            self.target_names = config.get("target_names")
            print(f"Dataset num_classes: {self.num_classes}")
            
        else:
            print("Invalid dataset name!")

        ###----ML-PARAMETERS-------###
        with open('./Hyperparameter Tuning/hyperparameters.json', 'r') as file:
            hyperparameters = json.load(file)

        self.lr_params = hyperparameters.get("lr_params", {})
        self.knn_params = hyperparameters.get("knn_params", {})
        self.gnb_params = hyperparameters.get("gnb_params", {})
        self.lin_svc_params = hyperparameters.get("lin_svc_params", {})
        self.dt_params = hyperparameters.get("dt_params", {})
        self.et_params = hyperparameters.get("et_params", {})
        self.xgb_params = hyperparameters.get("xgb_params", {})
        self.rf_params = hyperparameters.get("rf_params", {})
        self.dnn_params = hyperparameters.get("dnn_params", {})
        ##############################

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def get_filename_counter(self):
        self.filename_counter += 1
        return str(self.filename_counter) + ". "

    def printlog(self, message):
        message = str(message)
        print(message, flush=True)

    def get_ts(self):
        ts = time.time() - self.start_ts
        return f"{ts:>08.4f}"

    def save_model(self, model, name):
        Path(self.model_save_path).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(self.model_save_path, f"{self.model_save_version} {name}.pkl")
        # self.printlog(f"[{self.get_ts()}] Saving model to {filename}")
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, name):
        Path(self.model_save_path).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(self.model_save_path, f"{self.model_save_version} {name}.pkl")
        # self.printlog(f"[{self.get_ts()}] Loading model from {filename}")
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model

    def map_classes(self, df):
        df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: self.class_mapping.get(x, 'Others'))
        return df

    def pie_plot(self, df, cols_list, rows, cols):
        fig, axes = plt.subplots(rows, cols)
        for ax, col in zip(axes.ravel(), cols_list):
            df[col].value_counts().plot(ax=ax, kind='pie', figsize=(12, 8), fontsize=10, autopct='%1.0f%%')
            ax.set_title(str(col), fontsize=12)
        counter = self.get_filename_counter()
        plt.savefig(os.path.join(self.output_dir, f"{counter}{col}_pie_chart.png"))  # Save the chart
        # print(f"[{self.get_ts()}] Saved results to {self.output_dir}/{counter}{col}_pie_chart.png", flush=True)
        if (self.display_results): plt.show()
        plt.clf()

    def cm_plot(self, y_test, y_predict, name):
        # Combine all non-"Normal" classes into a single "Attack" class
            # Flatten CM attack classes into one
        y_test_combined = np.where(y_test == 0, 0, 1)
        test_predict_combined = np.where(y_predict == 0, 0, 1)
        cm = metrics.confusion_matrix(y_test_combined, test_predict_combined)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Attack'])
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.grid(False)
        cm_display.plot(ax=ax)
        counter = self.get_filename_counter()
        plt.savefig(os.path.join(self.output_dir, f"{counter}{name}_confusion_matrix.png"))
        # print(f"[{self.get_ts()}] Saved results to {self.output_dir}/{counter}{name}_confusion_matrix.png", flush=True)
        plt.clf()
        
        try:
            cm=metrics.confusion_matrix(y_test, y_predict)
            f,ax=plt.subplots(figsize=(5,5))
            sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
            plt.xlabel("y_pred")
            plt.ylabel("y_true")
            # plt.show()
            plt.savefig(os.path.join(self.output_dir, f"{counter}{name}_multi_confusion_matrix.png"))
            # print(f"[{self.get_ts()}] Saved results to {self.output_dir}/{counter}{name}_multi_confusion_matrix.png", flush=True)
            plt.clf()
        except Exception as e:
            print(e)
            print(f"Unable to plot CM for {name}")

    def preprocess(self, dataframe):
        # dataframe = dataframe.drop(self.drop_cols, axis=1)
        # df_num = dataframe.drop(self.cat_cols, axis=1)
        # num_cols = df_num.select_dtypes(include=[np.number]).columns
        # dataframe = pd.get_dummies(dataframe, columns=self.obj_cols)
        # df_num = dataframe[num_cols]
        # labels = dataframe[self.label_header]
        
        # # Replace NaN values with 0
        # df_num.fillna(0, inplace=True)
        
        # # Replace infinity values with 0
        # df_num.replace([np.inf, -np.inf], 0, inplace=True)
        # dataframe = pd.concat([df_num, labels], axis=1)

        dataframe = dataframe.drop(self.drop_cols, axis=1)
        # Replace NaN values with 0
        dataframe.fillna(0, inplace=True)
        
        # Replace infinity values with 0
        dataframe.replace([np.inf, -np.inf], 0, inplace=True)
        return dataframe

    def save_metrics_to_csv(self, eval_metrics, filepath):
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

    def evaluate_classification(self, model, name, x_test, y_test):
        self.printlog(f"[{self.get_ts()}] Evaluating classifier: {name}...")
        start_time = time.time()
        try:
            test_predict = model.predict(x_test)
            test_accuracy = metrics.accuracy_score(y_test, test_predict)
            test_precision = metrics.precision_score(y_test, test_predict, average=self.eval_average)
            test_recall = metrics.recall_score(y_test, test_predict, average=self.eval_average)
            test_f1 = metrics.f1_score(y_test, test_predict, average=self.eval_average)

            report = f"Evaluation Results for {name}: \n" + str(metrics.classification_report(y_test, test_predict, target_names=self.target_names))
            
        except Exception as e:
            traceback.print_exc()
            self.printlog(f"An error occurred while attempting to evaluate model: {name}")

        test_time = time.time() - start_time
        test_time_str = f"[{self.get_ts()}] Testing time: {test_time:.4f}"
        report = report + "\n" + test_time_str
        # self.printlog(test_time_str)
        eval_metrics = [str(name), test_accuracy, test_precision, test_recall, test_f1]
        self.cm_plot(y_test, test_predict, name)
        return test_predict, report, eval_metrics, test_time

    def f_importances(self, coef, names, top=-1, title="untitled"):
        imp = coef
        imp, names = zip(*sorted(list(zip(imp, names))))

        # Show all features
        if top == -1:
            top = len(names)
        
        plt.figure(figsize=(10, 10))
        plt.barh(range(top), imp[::-1][0:top], align='center')
        plt.yticks(range(top), names[::-1][0:top])
        plt.title(f'Feature importances for {title}')
        counter = self.get_filename_counter()
        plt.savefig(os.path.join(self.output_dir, f"{counter}Feature importances for {title}.png"))
        # print(f"[{self.get_ts()}] Saved results to {self.output_dir}/{counter}Feature importances for {title}.png", flush=True)
        if (self.display_results): plt.show()
        plt.clf()

    def run_lr(self, x_train, y_train, x_test, y_test):
        file_name = "Logistic Regression"
        if self.load_saved_models:
            lr = self.load_model(file_name)
        else:
            self.printlog(f"[{self.get_ts()}] Preparing Logistic Regression")
            start_time = time.time()
            lr = LogisticRegression(**self.lr_params).fit(x_train, y_train)
            train_time = time.time() - start_time
            train_time_str = f"[{self.get_ts()}] Training time: {train_time:.4f}"
        _, report, eval_metrics, test_time = self.evaluate_classification(lr, file_name, x_test, y_test)
        eval_metrics.append(train_time)
        eval_metrics.append(test_time)
        report = train_time_str + "\n" + report
        self.printlog(report)
        if self.save_trained_models: self.save_model(lr, file_name)
        return report, eval_metrics

    def run_knn(self, x_train, y_train, x_test, y_test):
        file_name = "KNeighborsClassifier"
        if self.load_saved_models:
            knn = self.load_model(file_name)
        else:
            self.printlog(f"[{self.get_ts()}] Preparing KNeighborsClassifier")
            start_time = time.time()
            knn = KNeighborsClassifier(**self.knn_params).fit(x_train, y_train)
            train_time = time.time() - start_time
            train_time_str = f"[{self.get_ts()}] Training time: {train_time:.4f}"
            # self.printlog(train_time_str)
        _, report, eval_metrics, test_time = self.evaluate_classification(knn, file_name, x_test, y_test)
        eval_metrics.append(train_time)
        eval_metrics.append(test_time)
        report = train_time_str + "\n" + report
        self.printlog(report)
        if self.save_trained_models: self.save_model(knn, file_name)
        return report, eval_metrics

    def run_gnb(self, x_train, y_train, x_test, y_test):
        file_name = "GaussianNB"
        if self.load_saved_models:
            gnb = self.load_model(file_name)
        else:
            self.printlog(f"[{self.get_ts()}] Preparing GaussianNB")
            start_time = time.time()
            gnb = GaussianNB(**self.gnb_params).fit(x_train, y_train)
            train_time = time.time() - start_time
            train_time_str = f"[{self.get_ts()}] Training time: {train_time:.4f}"
            # self.printlog(train_time_str)
        _, report, eval_metrics, test_time = self.evaluate_classification(gnb, file_name, x_test, y_test)
        eval_metrics.append(train_time)
        eval_metrics.append(test_time)
        report = train_time_str + "\n" + report
        self.printlog(report)
        if self.save_trained_models: self.save_model(gnb, file_name)
        return report, eval_metrics

    def run_lin_svc(self, x_train, y_train, x_test, y_test): 
        file_name = "Linear SVC(LBasedImpl)"
        if self.load_saved_models:
            lin_svc = self.load_model(file_name)
        else:
            self.printlog(f"[{self.get_ts()}] Preparing Linear SVC(LBasedImpl)")
            start_time = time.time()
            lin_svc = svm.LinearSVC(**self.lin_svc_params).fit(x_train, y_train)
            train_time = time.time() - start_time
            train_time_str = f"[{self.get_ts()}] Training time: {train_time:.4f}"
            # self.printlog(train_time_str)
        _, report, eval_metrics, test_time = self.evaluate_classification(lin_svc, file_name, x_test, y_test)
        eval_metrics.append(train_time)
        eval_metrics.append(test_time)
        report = train_time_str + "\n" + report
        self.printlog(report)
        if self.save_trained_models: self.save_model(lin_svc, file_name)
        return report, eval_metrics

    def run_dt(self, x_train, y_train, x_test, y_test):
        file_name = "DecisionTreeClassifier"
        if self.load_saved_models:
            dt = self.load_model(file_name)
            tdt = self.load_model(file_name)
        else:
            self.printlog(f"[{self.get_ts()}] Preparing Decision Tree")
            start_time = time.time()
            dt = DecisionTreeClassifier(**self.dt_params).fit(x_train, y_train)
            tdt = DecisionTreeClassifier(**self.dt_params).fit(x_train, y_train)
            train_time = time.time() - start_time
            train_time_str = f"[{self.get_ts()}] Training time: {train_time:.4f}"
            # self.printlog(train_time_str)
        _, report, eval_metrics, test_time = self.evaluate_classification(tdt, file_name, x_test, y_test)
        eval_metrics.append(train_time)
        eval_metrics.append(test_time)
        report = train_time_str + "\n" + report
        self.printlog(report)
        if self.save_trained_models: self.save_model(tdt, file_name)
        features_names = self.feature_names
        self.f_importances(abs(tdt.feature_importances_), features_names, top=18, title="Decision Tree")

        # print(f"[{self.get_ts()}] Generating results...", flush=True)
        fig = plt.figure(figsize=(60, 40))
        tree.plot_tree(dt, filled=True, feature_names=features_names.columns, fontsize=8)
        counter = self.get_filename_counter()
        plt.savefig(os.path.join(self.output_dir, f"{counter}Decision_tree.png"))
        # print(f"[{self.get_ts()}] Saved results to {self.output_dir}/{counter}Decision_tree.png", flush=True)
        plt.clf()
        return report, eval_metrics

    def run_rf(self, x_train, y_train, x_test, y_test):
        file_name = "RandomForestClassifier"
        if self.load_saved_models:
            rf = self.load_model(file_name)
        else:
            self.printlog(f"[{self.get_ts()}] Preparing RandomForest")
            start_time = time.time()
            rf = RandomForestClassifier(**self.rf_params).fit(x_train, y_train)
            train_time = time.time() - start_time
            train_time_str = f"[{self.get_ts()}] Training time: {train_time:.4f}"
            # self.printlog(train_time_str)
        _, report, eval_metrics, test_time = self.evaluate_classification(rf, file_name, x_test, y_test)
        eval_metrics.append(train_time)
        eval_metrics.append(test_time)
        report = train_time_str + "\n" + report
        self.printlog(report)
        features_names = self.feature_names
        self.f_importances(abs(rf.feature_importances_), features_names, top=18, title="Random Forest")
        if self.save_trained_models: self.save_model(rf, file_name)
        return report, eval_metrics

    def run_xgb(self, x_train, y_train, x_test, y_test):
        file_name = "XGBoost"
        if self.load_saved_models:
            xg_c = self.load_model(file_name)
        else:
            self.printlog(f"[{self.get_ts()}] Preparing XGBoost")
            start_time = time.time()
            xg_c = xgb.XGBClassifier(**self.xgb_params).fit(x_train, y_train)
            train_time = time.time() - start_time
            train_time_str = f"[{self.get_ts()}] Training time: {train_time:.4f}"
            # self.printlog(train_time_str)
        _, report, eval_metrics, test_time = self.evaluate_classification(xg_c, file_name, x_test, y_test)
        eval_metrics.append(train_time)
        eval_metrics.append(test_time)
        report = train_time_str + "\n" + report
        self.printlog(report)
        if self.save_trained_models: self.save_model(xg_c, file_name)
        return report, eval_metrics

    def run_et(self, x_train, y_train, x_test, y_test):
        file_name = "ExtraTrees"
        if self.load_saved_models:
            et = self.load_model(file_name)
        else:
            self.printlog(f"[{self.get_ts()}] Preparing ExtraTrees")
            start_time = time.time()
            et = ExtraTreesClassifier(**self.et_params).fit(x_train, y_train)
            train_time = time.time() - start_time
            train_time_str = f"[{self.get_ts()}] Training time: {train_time:.4f}"
            # self.printlog(train_time_str)
        _, report, eval_metrics, test_time = self.evaluate_classification(et, file_name, x_test, y_test)
        eval_metrics.append(train_time)
        eval_metrics.append(test_time)
        report = train_time_str + "\n" + report
        self.printlog(report)
        if self.save_trained_models: self.save_model(et, file_name)
        return report, eval_metrics

    def run_dnn(self, x_train, y_train, x_test, y_test):
        file_name = "Deep Neural Network"
        if self.load_saved_models:
            dnn = self.load_model(file_name)
        else:
            self.printlog(f"[{self.get_ts()}] Preparing DNN")
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=self.num_classes)
            y_test = tf.keras.utils.to_categorical(y_test, num_classes=self.num_classes)
            start_time = time.time()
            dnn = tf.keras.Sequential()
            for units in self.dnn_params["dense_layers"]:
                dnn.add(tf.keras.layers.Dense(units, activation=self.dnn_params["activation"], input_shape=(x_train.shape[1],)))
                dnn.add(tf.keras.layers.Dropout(self.dnn_params["dropout_rate"]))
            # Output Dense layer
            dnn.add(tf.keras.layers.Dense(self.num_classes, activation=self.dnn_params["output_activation"]))
            dnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.dnn_params["learning_rate"]), loss=self.dnn_params["loss"], metrics=self.dnn_params["metrics"])
            results = dnn.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))
            train_time = time.time() - start_time
            train_time_str = f"[{self.get_ts()}] Training time: {train_time:.4f}"
            # self.printlog(train_time_str)
            plt.subplots(figsize=(8, 6))
            plt.plot(results.history['accuracy'], label='Training Accuracy')
            plt.plot(results.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(loc='lower right')
            counter = self.get_filename_counter()
            plt.savefig(os.path.join(self.output_dir, f"{counter}Deep Neural Network.png"))
            # print(f"[{self.get_ts()}] Saved results to {self.output_dir}/{counter}Deep Neural Network.png", flush=True)
            if(self.display_results): plt.show()
            plt.clf()

        loss, accuracy = dnn.evaluate(x_test, y_test)
        self.printlog(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
        if self.save_trained_models: self.save_model(dnn, file_name)
        try:
            actual = y_test
            # predicted = dnn.predict(x_test)
            _, report, eval_metrics, test_time = self.evaluate_classification(dnn, file_name, x_test, y_test)
            eval_metrics.append(train_time)
            eval_metrics.append(test_time)
            report = train_time_str + "\n" + report
            self.printlog(report)
            return report, eval_metrics
        except Exception as e:
            print(e)
            traceback.print_exc()
            report = f"\nTest Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}"
            return report, []


    def model_runner(self, model_func, x_train, y_train, x_test, y_test):
        return model_func(x_train, y_train, x_test, y_test)
        
    # def run_models(self, x_train, y_train, x_test, y_test):
    #     if self.bool_gnb: self.run_gnb(x_train, y_train, x_test, y_test)
    #     if self.bool_xgb: self.run_xgb(x_train, y_train, x_test, y_test)
    #     if self.bool_et: self.run_et(x_train, y_train, x_test, y_test)
    #     if self.bool_dt: self.run_dt(x_train, y_train, x_test, y_test)
    #     if self.bool_rf: self.run_rf(x_train, y_train, x_test, y_test)
    #     if self.bool_lr: self.run_lr(x_train, y_train, x_test, y_test)
    #     if self.bool_lin_svc: self.run_lin_svc(x_train, y_train, x_test, y_test)
    #     if self.bool_knn: self.run_knn(x_train, y_train, x_test, y_test)
    #     if self.bool_dnn: self.run_dnn(x_train, y_train, x_test, y_test)

    def run_models(self, x_train, y_train, x_test, y_test):
        reports = []
        total_eval_metrics = []
        model_funcs = []
        
        # knn and xgb uses a lot of processing power, so run them individually
        if self.bool_xgb: 
            report, eval_metrics = self.run_xgb(x_train, y_train, x_test, y_test)
            reports.append(report)
            total_eval_metrics.append(eval_metrics)
        if self.bool_knn: 
            report, eval_metrics = self.run_knn(x_train, y_train, x_test, y_test)
            reports.append(report)
            total_eval_metrics.append(eval_metrics)
        
        # these models rarely use more than 1 thread, can run them in parallel
        # if self.bool_knn: model_funcs.append(self.run_knn)
        # if self.bool_xgb: model_funcs.append(self.run_xgb)
        if self.bool_gnb: model_funcs.append(self.run_gnb)
        if self.bool_et: model_funcs.append(self.run_et)
        if self.bool_dt: model_funcs.append(self.run_dt)
        if self.bool_rf: model_funcs.append(self.run_rf)
        if self.bool_lr: model_funcs.append(self.run_lr)
        if self.bool_lin_svc: model_funcs.append(self.run_lin_svc)
        if self.bool_dnn: model_funcs.append(self.run_dnn)

        # Use ProcessPoolExecutor to run models in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(self.max_workers, len(model_funcs))) as executor:
            futures = [executor.submit(self.model_runner, func, x_train, y_train, x_test, y_test) for func in model_funcs]
            for future in concurrent.futures.as_completed(futures):
                try:
                    report, eval_metrics = future.result()
                    reports.append(report)
                    total_eval_metrics.append(eval_metrics)
                except Exception as exc:
                    print(f'[{self.get_ts()}] Generated an exception: {exc}')
                    traceback.print_exc()
        
        # Combine all reports into a single string
        try:
            combined_report = "\n\n".join(reports) if reports else "No reports generated."
        except Exception as e:
            print(f"[{self.get_ts()}] Unable to generate report.txt: {e}")

        # Write the combined report to a file
        with open(f'{self.output_dir}/log.txt', 'w') as report_file:
            report_file.write(combined_report)
        
        filepath = f"{self.output_dir}/results.csv"
        with open(filepath, 'w', newline='') as csvfile:
            # Create a CSV writer object
            csv_writer = csv.writer(csvfile)

            # Write the header row
            header = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'Train Time', 'Test Time']
            csv_writer.writerow(header)

            # Write metrics for each model
            for metrics in total_eval_metrics:
                row_data = metrics
                csv_writer.writerow(row_data)

            print(f"Metrics saved to {filepath}")

    def start(self):
        warnings.filterwarnings('ignore')

        self.printlog(f"[{self.get_ts()}] Init complete!")
        self.printlog(f"[{self.get_ts()}] Reading from {self.train_path}")

        # Read Train and Test dataset
        data_train = pd.read_csv(self.train_path)
        if not self.use_single_dataset: 
            data_test = pd.read_csv(self.test_path)
        else: 
            data_test = data_train
            
        if (self.read_cols_from_csv): 
            # Removes white spaces
            columns = data_train.columns.str.strip().tolist()

        # Assign names for columns
        data_train.columns = columns
        data_test.columns = columns
        data_train.info()
        if not self.use_single_dataset: data_test.info()

        self.printlog(f"[{self.get_ts()}] Mapping outcomes...")

        if self.generate_statistics_pie:
            for i in self.pie_stats:
                self.pie_plot(data_train, i, 1, 2)

        # Process and split dataset
        if self.use_single_dataset: 
            data_train = self.map_classes(data_train)
            label_mapping = self.class_mapping
            
            scaled_train = self.preprocess(data_train)

            x = scaled_train.drop(self.label_header , axis = 1).values
            y = scaled_train[self.label_header].values

            value_counts = data_train[self.label_header].value_counts()
            self.printlog(value_counts)
            
            self.feature_names = data_train.drop(self.label_header, axis = 1)

            y=np.ravel(y)
            y = y.astype('int')

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.split_test_ratio, random_state=self.rndm_state)

            # SMOTE
            smote = SMOTE(random_state=42)
            x_train, y_train = smote.fit_resample(x_train, y_train)
            self.run_models(x_train, y_train, x_test, y_test)

        # Evaluate using separate train and test datasets
        else:
            data_train = self.map_classes(data_train)
            label_mapping = self.class_mapping
            
            scaled_train = self.preprocess(data_train)
            x_train = scaled_train.drop(self.label_header, axis=1).values
            y_train = scaled_train[self.label_header].values
            y_train = y_train.astype('int')

            # Process testing set
            scaled_test = self.preprocess(data_test)
            x_test = scaled_test.drop(self.label_header, axis=1).values
            y_test = scaled_test[self.label_header].values
            y_test = y_test.astype('int')

            self.run_models(x_train, y_train, x_test, y_test)

        # Save all metrics to file
        # self.save_metrics_to_csv(self.kernal_evals, f"{self.output_dir}/results.csv")
        print(f"[{self.get_ts()}] End of program")

#-------------------------------- MAIN --------------------------------
if __name__ == "__main__":
    job = Ml()
    job.start()