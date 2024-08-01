from networkx import dfs_predecessors
import pandas as pd
df = pd.read_csv('./input/CICIDS2017/CICIDS2017.csv')
df.columns = df.columns.str.strip().tolist()
label_counts = df['Label'].value_counts()
print(label_counts)


# Randomly sample instances from majority classes
df_other = df[~((df['Label']=='BENIGN')|(df['Label']=='DoS Hulk')|(df['Label']=='PortScan')|(df['Label']=='DDoS'))]
df_BENIGN = df[df['Label'] == 'BENIGN'].sample(frac=0.05)
df_dos = df[df['Label'] == 'DoS Hulk'].sample(frac=0.1)
df_portscan = df[df['Label'] == 'PortScan'].sample(frac=0.1)
df_ddos = df[df['Label'] == 'DDoS'].sample(frac=0.1)
df_s = pd.concat([df_BENIGN, df_dos, df_portscan, df_ddos, df_other])



# df_minor = df[(df['Label']=='WebAttack')|(df['Label']=='Bot')|(df['Label']=='Infiltration')]
# df_BENIGN = df[(df['Label']=='BENIGN')].sample(n=None, frac=0.01, replace=False, weights=None, random_state=None, axis=0)
# df_DoS = df[(df['Label']=='DoS')].sample(n=None, frac=0.05, replace=False, weights=None, random_state=None, axis=0)
# df_PortScan = df[(df['Label']=='PortScan')].sample(n=None, frac=0.05, replace=False, weights=None, random_state=None, axis=0)
# df_BruteForce = df[(df['Label']=='BruteForce')].sample(n=None, frac=0.2, replace=False, weights=None, random_state=None, axis=0)

# df_s = df_BENIGN.append(df_DoS).append(df_PortScan).append(df_BruteForce).append(df_minor)
df_s = df_s.sort_index()
label_counts_s = df_s['Label'].value_counts()
print(label_counts_s)

# Save the sampled dataset
df_s.to_csv('./input/CICIDS2017/small-CICIDS2017.csv',index=0)