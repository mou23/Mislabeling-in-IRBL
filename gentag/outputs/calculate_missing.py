import pandas as pd
import sys

file1 = sys.argv[1]
file2 = sys.argv[2]

df1 = pd.read_csv(file1, header=None, names=['Project', 'BugID', 'Label', 'Predicted', 'Explanation'])
df2 = pd.read_csv(file2, header=None, names=['Project', 'BugID', 'Label', 'Predicted', 'Explanation'])
# df2 = pd.read_csv(file2, header=None, names=['Project', 'BugID', 'Label', 'Predicted', 'Explanation'])

# Find bug IDs that are in one set but not the other
missing_in_df2 = df1[~df1['BugID'].isin(df2['BugID'])]
missing_in_df1 = df2[~df2['BugID'].isin(df1['BugID'])]

print(f"Bug IDs in {file1} but missing in {file2}:")
print(missing_in_df2[['Project', 'BugID']].to_string(index=False))
print(f"\nTotal missing in {file2}: {len(missing_in_df2)}")

print(f"\nBug IDs in {file2} but missing in {file1}:")
print(missing_in_df1[['Project', 'BugID']].to_string(index=False))
print(f"\nTotal missing in {file1}: {len(missing_in_df1)}")

# Alternative: Show all unique bug IDs from both files
all_bug_ids_df1 = set(df1['BugID'])
all_bug_ids_df2 = set(df2['BugID'])

print(f"\nTotal unique bug IDs in {file1}: {len(all_bug_ids_df1)}")
print(f"Total unique bug IDs in {file2}: {len(all_bug_ids_df2)}")
print(f"Bug IDs only in {file1}: {sorted(all_bug_ids_df1 - all_bug_ids_df2)}")
print(f"Bug IDs only in {file2}: {sorted(all_bug_ids_df2 - all_bug_ids_df1)}")

# Check for duplicates within each file (using Project + BugID as primary key)
duplicates_df1 = df1[df1.duplicated(subset=['Project', 'BugID'], keep=False)]
duplicates_df2 = df2[df2.duplicated(subset=['Project', 'BugID'], keep=False)]

print(f"Duplicates in {file1}: {len(duplicates_df1)}")
print(f"Duplicates in {file2}: {len(duplicates_df2)}")

# Check for duplicates across both files (same Project + BugID appearing in both)
common_keys = set(zip(df1['Project'], df1['BugID'])).intersection(set(zip(df2['Project'], df2['BugID'])))
print(f"Project+BugID combinations appearing in both files: {len(common_keys)}")
