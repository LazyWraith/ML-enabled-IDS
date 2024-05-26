import pandas as pd
from sklearn.preprocessing import LabelEncoder

in_file = './input/CICIDS2017/small-CICIDS2017.csv'
out_file = './input/CICIDS2017/small-CICIDS2017-remapped.csv'

df = pd.read_csv(in_file)
df.columns = df.columns.str.strip().tolist()
print(df['Label'].value_counts())

original_classes = [
    'BENIGN', 'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye', 'FTP-Patator', 
    'SSH-Patator', 'DoS slowloris', 'DoS Slowhttptest', 'Bot', 
    'Web Attack - Brute Force', 'Web Attack - XSS', 'Infiltration', 
    'Web Attack - Sql Injection', 'Heartbleed'
]

# Define the new class mapping
class_mapping = {
    'BENIGN': 'BENIGN',
    'Bot': 'Bot',
    'DDoS': 'DoS',
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'PortScan': 'PortScan',
    'FTP-Patator': 'BruteForce',
    'SSH-Patator': 'BruteForce',
    'Web Attack � Brute Force': 'Web Attack',
    'Web Attack � XSS': 'Web Attack',
    'Web Attack � Sql Injection': 'Web Attack',
    'Infiltration': 'Others',
    'Heartbleed': 'Others'
}

# Map the original classes to new classes, defaulting to 'Others' for any undefined classes
df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: class_mapping.get(x, 'Others'))

# Encode the new classes into integers
labelencoder = LabelEncoder()
df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])

# Create the new label mapping dictionary
new_label_mapping = {index: label for index, label in enumerate(labelencoder.classes_)}

# Output the new label mapping
print(new_label_mapping)
print(df['Label'].value_counts())

# df.to_csv(out_file)