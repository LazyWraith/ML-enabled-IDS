import pandas as pd

in_file = './Captures/PortScan/port-scan.csv'
out_file = './Captures/PortScan/cic_port-scan.csv'
label = 'PortScan'

# Read the source CSV file
df = pd.read_csv(in_file)

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
    'idle_min' : 'Idle Min'
}

# Drop columns that are not specified in the column mappings
df = df[list(column_mapping.keys())]

# Rename columns
df.rename(columns=column_mapping, inplace=True)

# Add Label column
df['Label'] = label

# Write result to a new CSV file
df.to_csv(out_file, index=False)
