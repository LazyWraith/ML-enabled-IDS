import pandas as pd
import glob

# Specify the path to your CSV files
files_path = './input/CICIDS2017/*.csv'

# Get a list of file names that match the pattern
files = glob.glob(files_path)
print(files)

# Initialize an empty DataFrame to store the concatenated data
concatenated_data = pd.DataFrame()

# Loop through each file and concatenate the data
for file in files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)
    
    # Concatenate the DataFrame to the existing data
    concatenated_data = pd.concat([concatenated_data, df], ignore_index=True)

# Write the concatenated data to a new CSV file
concatenated_data.to_csv('./input/CICIDS2017/CICIDS2017.csv', index=False)
