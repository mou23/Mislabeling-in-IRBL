import pandas as pd
import sys

result_file_1 = sys.argv[1]
result_file_2 = sys.argv[2]
result_file_3 = sys.argv[3]

# Read the CSV files
df1 = pd.read_csv(result_file_1, header=None, names=['Project', 'BugID', 'Label', 'Predicted', 'Explanation'])
df2 = pd.read_csv(result_file_2, header=None, names=['Project', 'BugID', 'Label', 'Predicted', 'Explanation'])
df3 = pd.read_csv(result_file_3, header=None, names=['Project', 'BugID', 'Label', 'Predicted', 'Explanation'])

# Merge the dataframes on Project and BugID, starting with df3
merged_df = df3[['Project', 'BugID', 'Label', 'Predicted']].merge(
    df1[['Project', 'BugID', 'Predicted']], 
    on=['Project', 'BugID'], 
    suffixes=('_run2', '_run3')
).merge(
    df2[['Project', 'BugID', 'Predicted']], 
    on=['Project', 'BugID']
)

# Rename columns for clarity
merged_df.columns = ['Project', 'BugID', 'ground_truth', 'result1', 'result2', 'result3']

# Save to new CSV file
# merged_df.to_csv('compare.csv', index=False)

# Find rows where classifications don't match, but only for BUG ground truth
mismatches = merged_df[
    (merged_df['ground_truth'] == 'BUG') & (
        (merged_df['result1'] != merged_df['result2']) | 
        (merged_df['result2'] != merged_df['result3']) | 
        (merged_df['result3'] != merged_df['result1'])
    )
]

# Save mismatches to a separate file
# mismatches.to_csv('mismatches.csv', index=False)

print(f"Total rows: {len(merged_df)}")
print(f"Total BUG rows: {len(merged_df[merged_df['ground_truth'] == 'BUG'])}")
print(f"Number of mismatches in BUG rows: {len(mismatches)}")
print("\nMismatch patterns for BUG rows:")
print("run1 vs run2 vs run3:")
print(mismatches[['ground_truth', 'result1', 'result2', 'result3']].value_counts().head(10)) 