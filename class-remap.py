import pandas as pd

in_file = './input/CICIDS2017/small-CICIDS2017.csv'
out_file = './input/CICIDS2017/small-CICIDS2017-remapped.csv'

df = pd.read_csv(in_file)
df.columns = df.columns.str.strip().tolist()
print(df['Label'].value_counts())

# Define attack class mapping for CICIDS2017
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

print(df['Label'].value_counts())

df.to_csv(out_file)