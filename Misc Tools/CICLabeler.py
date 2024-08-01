import pandas as pd
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Process some files.')

# Add the arguments
parser.add_argument('-i', '--in_file', type=str, required=True, help='The input file')
parser.add_argument('-o', '--out_file', type=str, required=True, help='The output file')
parser.add_argument('-l', '--label', type=str, required=False, help='Add class labels')

# Parse the arguments
args = parser.parse_args()

# Read from the input file
in_file = args.in_file

# Write to the output file
out_file = args.out_file

# in_file = "C:/Users/ethyr/Desktop/ML-enabled-IDS/Captures/UOWM_IEC104_Dataset/20200429_UOWM_IEC104_Dataset_c_sc_na_1_DoS/20200429_UOWM_IEC104_Dataset_c_sc_na_1_DoS_qtester/20200429_UOWM_IEC104_Dataset_c_sc_na_1_DoS_qtester.pcap_Flow.csv"
# out_file = './Captures/UOWM_IEC104_Dataset/converted/20200429_UOWM_IEC104_Dataset_c_sc_na_1_DoS_qtester.pcap_Flow.csv'
# label = 'Others'

# Read the source CSV file
df = pd.read_csv(in_file)

benign_label = 'BENIGN'
attack_label = 'PortScan'

# Attack devices IP
label_mapping = {
    '192.168.1.20' : str(attack_label),
    '192.168.1.21' : str(attack_label),
    '192.168.1.22' : str(attack_label),
    '192.168.1.23' : str(attack_label),
    '192.168.1.24' : str(attack_label),
    '192.168.1.25' : str(attack_label),
    '192.168.1.26' : str(attack_label),
    '192.168.1.27' : str(attack_label),
    '192.168.1.28' : str(attack_label),
    '192.168.1.29' : str(attack_label),
    '192.168.1.30' : str(attack_label),
    '192.168.1.31' : str(attack_label),
    '192.168.1.32' : str(attack_label),
    '192.168.1.33' : str(attack_label),
    '192.168.1.34' : str(attack_label),
    '192.168.1.35' : str(attack_label),
    '192.168.1.36' : str(attack_label),
    '192.168.1.37' : str(attack_label),
    '192.168.1.38' : str(attack_label),
    '192.168.1.39' : str(attack_label)
}

df['Label'] = str(benign_label)
df['Label'] = df['src_ip'].apply(lambda ip: label_mapping.get(ip, str(benign_label)))



# Dictionary mapping source column names to target column names
column_mapping = {
    'dst_port' : 'Destination Port',
    'flow_duration' : 'Flow Duration',
    'tot_fwd_pkts' : 'Total Fwd Packets',
    'tot_bwd_pkts' : 'Total Backward Packets',
    'totlen_fwd_pkts' : 'Total Length of Fwd Packets',
    'totlen_bwd_pkts' : 'Total Length of Bwd Packets',
    'fwd_pkt_len_max' : 'Fwd Packet Length Max',
    'fwd_pkt_len_min' : 'Fwd Packet Length Min',
    'fwd_pkt_len_mean' : 'Fwd Packet Length Mean',
    'fwd_pkt_len_std' : 'Fwd Packet Length Std',
    'bwd_pkt_len_max' : 'Bwd Packet Length Max',
    'bwd_pkt_len_min' : 'Bwd Packet Length Min',
    'bwd_pkt_len_mean' : 'Bwd Packet Length Mean',
    'bwd_pkt_len_std' : 'Bwd Packet Length Std',
    'flow_byts_s' : 'Flow Bytes/s',
    'flow_pkts_s' : 'Flow Packets/s',
    'flow_iat_mean' : 'Flow IAT Mean',
    'flow_iat_std' : 'Flow IAT Std',
    'flow_iat_max' : 'Flow IAT Max',
    'flow_iat_min' : 'Flow IAT Min',
    'fwd_iat_tot' : 'Fwd IAT Total',
    'fwd_iat_mean' : 'Fwd IAT Mean',
    'fwd_iat_std' : 'Fwd IAT Std',
    'fwd_iat_max' : 'Fwd IAT Max',
    'fwd_iat_min' : 'Fwd IAT Min',
    'bwd_iat_tot' : 'Bwd IAT Total',
    'bwd_iat_mean' : 'Bwd IAT Mean',
    'bwd_iat_std' : 'Bwd IAT Std',
    'bwd_iat_max' : 'Bwd IAT Max',
    'bwd_iat_min' : 'Bwd IAT Min',
    'fwd_psh_flags' : 'Fwd PSH Flags',
    'bwd_psh_flags' : 'Bwd PSH Flags',
    'fwd_urg_flags' : 'Fwd URG Flags',
    'bwd_urg_flags' : 'Bwd URG Flags',
    'fwd_header_len' : 'Fwd Header Length',
    'bwd_header_len' : 'Bwd Header Length',
    'fwd_pkts_s' : 'Fwd Packets/s',
    'bwd_pkts_s' : 'Bwd Packets/s',
    'pkt_len_min' : 'Min Packet Length',
    'pkt_len_max' : 'Max Packet Length',
    'pkt_len_mean' : 'Packet Length Mean',
    'pkt_len_std' : 'Packet Length Std',
    'pkt_len_var' : 'Packet Length Variance',
    'fin_flag_cnt' : 'FIN Flag Count',
    'syn_flag_cnt' : 'SYN Flag Count',
    'rst_flag_cnt' : 'RST Flag Count',
    'psh_flag_cnt' : 'PSH Flag Count',
    'ack_flag_cnt' : 'ACK Flag Count',
    'urg_flag_cnt' : 'URG Flag Count',
    'cwe_flag_count' : 'CWE Flag Count',
    'ece_flag_cnt' : 'ECE Flag Count',
    'down_up_ratio' : 'Down/Up Ratio',
    'pkt_size_avg' : 'Average Packet Size',
    'fwd_seg_size_avg' : 'Avg Fwd Segment Size',
    'bwd_seg_size_avg' : 'Avg Bwd Segment Size',
    'fwd_byts_b_avg' : 'Fwd Avg Bytes/Bulk',
    'fwd_pkts_b_avg' : 'Fwd Avg Packets/Bulk',
    'fwd_blk_rate_avg' : 'Fwd Avg Bulk Rate',
    'bwd_byts_b_avg' : 'Bwd Avg Bytes/Bulk',
    'bwd_pkts_b_avg' : 'Bwd Avg Packets/Bulk',
    'bwd_blk_rate_avg' : 'Bwd Avg Bulk Rate',
    'subflow_fwd_pkts' : 'Subflow Fwd Packets',
    'subflow_fwd_byts' : 'Subflow Fwd Bytes',
    'subflow_bwd_pkts' : 'Subflow Bwd Packets',
    'subflow_bwd_byts' : 'Subflow Bwd Bytes',
    'init_fwd_win_byts' : 'Init_Win_bytes_forward',
    'init_bwd_win_byts' : 'Init_Win_bytes_backward',
    'fwd_act_data_pkts' : 'act_data_pkt_fwd',
    'fwd_seg_size_min' : 'min_seg_size_forward',
    'active_mean' : 'Active Mean',
    'active_std' : 'Active Std',
    'active_max' : 'Active Max',
    'active_min' : 'Active Min',
    'idle_mean' : 'Idle Mean',
    'idle_std' : 'Idle Std',
    'idle_max' : 'Idle Max',
    'idle_min' : 'Idle Min',
    "Label": "Label"
}

column_mapping_iec = {
    "Dst IP": "Destination Port",
    "Flow Duration": "Flow Duration",
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Flow IAT Mean": "Flow IAT Mean",
    "Flow IAT Std": "Flow IAT Std",
    "Flow IAT Max": "Flow IAT Max",
    "Flow IAT Min": "Flow IAT Min",
    "Fwd IAT Tot": "Fwd IAT Total",
    "Fwd IAT Mean": "Fwd IAT Mean",
    "Fwd IAT Std": "Fwd IAT Std",
    "Fwd IAT Max": "Fwd IAT Max",
    "Fwd IAT Min": "Fwd IAT Min",
    "Bwd IAT Tot": "Bwd IAT Total",
    "Bwd IAT Mean": "Bwd IAT Mean",
    "Bwd IAT Std": "Bwd IAT Std",
    "Bwd IAT Max": "Bwd IAT Max",
    "Bwd IAT Min": "Bwd IAT Min",
    "Fwd PSH Flags": "Fwd PSH Flags",
    "Bwd PSH Flags": "Bwd PSH Flags",
    "Fwd URG Flags": "Fwd URG Flags",
    "Bwd URG Flags": "Bwd URG Flags",
    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Len Min": "Min Packet Length",
    "Pkt Len Max": "Max Packet Length",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",
    "FIN Flag Cnt": "FIN Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "URG Flag Cnt": "URG Flag Count",
    "CWE Flag Count": "CWE Flag Count",
    "ECE Flag Cnt": "ECE Flag Count",
    "Down/Up Ratio": "Down/Up Ratio",
    "Pkt Size Avg": "Average Packet Size",
    "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    "Fwd Byts/b Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Pkts/b Avg": "Fwd Avg Packets/Bulk",
    "Fwd Blk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Byts/b Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Pkts/b Avg": "Bwd Avg Packets/Bulk",
    "Bwd Blk Rate Avg": "Bwd Avg Bulk Rate",
    "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",
    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
    "Active Mean": "Active Mean",
    "Active Std": "Active Std",
    "Active Max": "Active Max",
    "Active Min": "Active Min",
    "Idle Mean": "Idle Mean",
    "Idle Std": "Idle Std",
    "Idle Max": "Idle Max",
    "Idle Min": "Idle Min",
    "Label": "Label"
}


# Drop columns that are not specified in the column mappings
df = df[list(column_mapping.keys())]

# Rename columns
df.rename(columns=column_mapping, inplace=True)

if args.label:
    # Add Label column
    label = args.label
    df['Label'] = label

# Write result to a new CSV file
df.to_csv(out_file, index=False)
