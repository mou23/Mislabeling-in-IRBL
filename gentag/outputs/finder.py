import pandas as pd
import numpy as np

# Fix the column names by removing extra quotes
df_herzig = pd.read_csv("herzig-majority.csv", names=['project', 'id', 'label', 'predicted', 'explanation'])
df_herzig['actual'] = np.where(df_herzig['label'] == "BUG", "BUG", "NOT BUG")

df_setfit = pd.read_csv("../setfit/setfit_output.csv", names=['id', 'label', 'predicted'])
df_setfit['actual'] = np.where(df_setfit['label'] == "BUG", "BUG", "NOT BUG")
df_setfit['predicted'] = np.where(df_setfit['predicted'] == "BUG", "BUG", "NOT BUG")

df_colavito = pd.read_csv("../colavito/colavito-majority.csv", names=['project', 'id', 'label', 'predicted', 'explanation'])
df_colavito['actual'] = np.where(df_colavito['label'] == "BUG", "BUG", "NOT BUG")
df_colavito['predicted'] = np.where(df_colavito['predicted'] == "BUG", "BUG", "NOT BUG")

# Function to check if prediction is correct
def is_correct_prediction(row):
    return row['actual'] == row['predicted']

# Add correctness columns to each dataframe
df_herzig['herzig_correct'] = df_herzig.apply(is_correct_prediction, axis=1)
df_setfit['setfit_correct'] = df_setfit.apply(is_correct_prediction, axis=1)
df_colavito['colavito_correct'] = df_colavito.apply(is_correct_prediction, axis=1)

# print(df_herzig.head(40))

# Merge all three dataframes
merged_df = df_herzig.merge(df_setfit[['id', 'setfit_correct']], on='id', how='inner')
merged_df = merged_df.merge(df_colavito[['id', 'colavito_correct']], on='id', how='inner')

# Find cases where herzig was correct but setfit and colavito were wrong
target_cases = merged_df[
    (merged_df['herzig_correct'] == True) & 
    ((merged_df['setfit_correct'] == False) & 
    (merged_df['colavito_correct'] == False)) &
    (merged_df['label'] == "REFACTORING")
]

print(f"Found {len(target_cases)} cases where herzig was correct but setfit and colavito were wrong")
print("\nThese cases:")
print(target_cases[['project', 'id', 'label', 'predicted', 'explanation']].to_string(index=False))