import pandas as pd
import numpy as np
from collections import Counter
import sys
import os

def majority_vote(row):
    preds = [row['pred1'], row['pred2'], row['pred3']]
    counts = Counter(preds)
    most_common = counts.most_common(1)[0]
    return most_common[0] if most_common[1] > 1 else 'BUG' # or UNKNOWN for multi label classification

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python result_process_herzig_majority.py <file_path1> <file_path2> <file_path3>")
        sys.exit(1)

    file_path_1 = sys.argv[1]
    file_path_2 = sys.argv[2]
    file_path_3 = sys.argv[3]

    df1 = pd.read_csv(file_path_1, usecols=[1, 2, 3], names=['id', 'actual', 'predicted'], header=None)
    df2 = pd.read_csv(file_path_2, usecols=[1, 2, 3], names=['id', 'actual', 'predicted'], header=None)
    df3 = pd.read_csv(file_path_3, usecols=[1, 2, 3], names=['id', 'actual', 'predicted'], header=None)

    # Comment out to get multi-label classification
    df1['predicted'] = df1['predicted'].apply(lambda x: "BUG" if x == "BUG" else "NOT BUG")
    df2['predicted'] = df2['predicted'].apply(lambda x: "BUG" if x == "BUG" else "NOT BUG")
    df3['predicted'] = df3['predicted'].apply(lambda x: "BUG" if x == "BUG" else "NOT BUG")

    df1 = df1.sort_values(by=['id']).reset_index(drop=True)
    df2 = df2.sort_values(by=['id']).reset_index(drop=True)
    df3 = df3.sort_values(by=['id']).reset_index(drop=True)

    df = pd.DataFrame(columns=['project', 'id', 'actual', 'predicted', 'explanation'])
    df['id'] = df1['id']
    df['actual'] = df1['actual']
    df['pred1'] = df1['predicted']
    df['pred2'] = df2['predicted']
    df['pred3'] = df3['predicted']
    df['predicted'] = df.apply(majority_vote, axis=1)
    df = df[['project', 'id', 'actual', 'predicted', 'explanation']]

    df.to_csv('outputs/herzig-pure-majority.csv', index=False, header=False)