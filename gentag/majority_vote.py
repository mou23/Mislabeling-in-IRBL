import pandas as pd
import os
import sys
from collections import Counter

def majority_vote(row):
    preds = [row['pred1'], row['pred2'], row['pred3']]
    counts = Counter(preds)
    most_common = counts.most_common(1)[0]
    return most_common[0] if most_common[1] > 1 else 'BUG' # or UNKNOWN for multi label classification

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python result_process.py <csv_file_1> <csv_file_2> <csv_file_3>")
        exit()

    csv_path_1 = sys.argv[1]
    csv_path_2 = sys.argv[2]
    csv_path_3 = sys.argv[3]

    df1 = pd.read_csv(csv_path_1, header=None, names=['Project', 'BugID', 'Label', 'Predicted', 'Explanation'])
    df2 = pd.read_csv(csv_path_2, header=None, names=['Project', 'BugID', 'Label', 'Predicted', 'Explanation'])
    df3 = pd.read_csv(csv_path_3, header=None, names=['Project', 'BugID', 'Label', 'Predicted', 'Explanation'])

    # Comment out to get multi-label classification
    df1['Predicted'] = df1['Predicted'].apply(lambda x: "BUG" if x == "BUG" else "NOT BUG")
    df2['Predicted'] = df2['Predicted'].apply(lambda x: "BUG" if x == "BUG" else "NOT BUG")
    df3['Predicted'] = df3['Predicted'].apply(lambda x: "BUG" if x == "BUG" else "NOT BUG")

    df1 = df1.sort_values(by=['Project', 'BugID']).reset_index(drop=True)
    df2 = df2.sort_values(by=['Project', 'BugID']).reset_index(drop=True)
    df3 = df3.sort_values(by=['Project', 'BugID']).reset_index(drop=True)

    df = pd.DataFrame(columns=['Project', 'BugID', 'Label', 'Predicted', 'Explanation'])

    df['Project'] = df1['Project']
    df['BugID'] = df1['BugID']
    df['Label'] = df1['Label']
    df['pred1'] = df1['Predicted']
    df['pred2'] = df2['Predicted']
    df['pred3'] = df3['Predicted']
    df['Predicted'] = df.apply(majority_vote, axis=1)
    df = df[['Project', 'BugID', 'Label', 'Predicted']]
    df['Explanation'] = ''

    file_name = csv_path_1.split('-')[0] + '-majority.csv'
    df.to_csv(file_name, index=False, header=False)