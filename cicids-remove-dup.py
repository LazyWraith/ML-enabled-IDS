import pandas as pd
in_file = './input/CICIDS2017/small-CICIDS2017-remapped.csv'
out_file = './input/CICIDS2017/small-CICIDS2017-remapped-deduplicated.csv'
df = pd.read_csv(in_file)
df.drop(["Fwd Header Length.1"], axis = 1, inplace=True)
df.to_csv(out_file, index=False)