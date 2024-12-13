import pandas as pd
from glob import glob

file_pattern = "../dataset/Top10_Comments_*.csv"
output_path_sampled = "../dataset/GNN_subset.csv"

files = glob(file_pattern)
print(f"Files matched: {files}")
if not files:
    raise ValueError("No files found to process. Check the file path or naming pattern.")

all_dfs = []
for file_path in files:
    print(f"Loading file: {file_path}")
    try:
        df = pd.read_csv(file_path)
        if not df.empty:
            all_dfs.append(df)
        else:
            print(f"File {file_path} is empty and skipped.")
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")

if not all_dfs:
    raise ValueError("No valid data to concatenate. Check your files.")

combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"Combined dataset shape: {combined_df.shape}")

sampled_df = combined_df.sample(n=4000, random_state=42)
print(f"Sampled dataset shape: {sampled_df.shape}")

sampled_df.to_csv(output_path_sampled, index=False)
print(f"Sampled dataset saved: {output_path_sampled}")
