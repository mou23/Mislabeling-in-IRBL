import pandas as pd
import sys

result_file_1 = sys.argv[1]
result_file_2 = sys.argv[2]
result_file_3 = sys.argv[3]

# Read the CSV files
df1 = pd.read_csv(result_file_1, header=None, names=['Project', 'BugID', 'Label', 'Predicted', 'Explanation'])
df2 = pd.read_csv(result_file_2, header=None, names=['Project', 'BugID', 'Label', 'Predicted', 'Explanation'])
df3 = pd.read_csv(result_file_3, header=None, names=['Project', 'BugID', 'Label', 'Predicted', 'Explanation'])

# Convert any non-BUG predictions to "Not BUG"
for df in [df1, df2, df3]:
    df['Predicted'] = df['Predicted'].apply(lambda x: 'NOT BUG' if x != 'BUG' else x)


df1 = df1.sort_values(by=['Project', 'BugID']).reset_index(drop=True)
df2 = df2.sort_values(by=['Project', 'BugID']).reset_index(drop=True)
df3 = df3.sort_values(by=['Project', 'BugID']).reset_index(drop=True)

df_all = df1[['Project', 'BugID']].copy()
df_all['Pred1'] = df1['Predicted']
df_all['Pred2'] = df2['Predicted']
df_all['Pred3'] = df3['Predicted']

# Compute agreement info
def compute_agreement(row):
    preds = [row['Pred1'], row['Pred2'], row['Pred3']]
    unique_preds = set(preds)
    return pd.Series({
        'all_agree': len(unique_preds) == 1,
        'majority_agree': len(unique_preds) <= 2,
        'num_unique_preds': len(unique_preds),
    })

df_all = df_all.join(df_all.apply(compute_agreement, axis=1))

# Compute pairwise agreement rates
pairwise_12 = (df_all['Pred1'] == df_all['Pred2']).mean()
pairwise_13 = (df_all['Pred1'] == df_all['Pred3']).mean()
pairwise_23 = (df_all['Pred2'] == df_all['Pred3']).mean()

# Compute overall agreement stats
total = len(df_all)
all_agree_pct = df_all['all_agree'].mean() * 100
majority_agree_pct = df_all['majority_agree'].mean() * 100
avg_unique_preds = df_all['num_unique_preds'].mean()

# Print results
print("Pairwise Agreement Rates:")
print(f"Run1 vs Run2: {pairwise_12:.2%}")
print(f"Run1 vs Run3: {pairwise_13:.2%}")
print(f"Run2 vs Run3: {pairwise_23:.2%}")

print("\nOverall Prediction Consistency:")
print(f"All 3 runs agreed: {all_agree_pct:.2f}%")
print(f"At least 2 runs agreed (majority): {majority_agree_pct:.2f}%")
print(f"Avg. unique predictions per input: {avg_unique_preds:.2f}")