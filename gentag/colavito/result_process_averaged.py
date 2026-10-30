import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import sys

def binary_classification(df):
    df['actual'] = np.where(df['actual'] == "BUG", "BUG", "NOT BUG")
    df['predicted'] = np.where(df['predicted'] == "bug", "BUG", "NOT BUG")
    
    custom_mapping = {'BUG': 1, 'NOT BUG': 0}
    df['actual_encoded'] = df['actual'].map(custom_mapping)
    df['predicted_encoded'] = df['predicted'].map(custom_mapping)

    y_true = df['actual_encoded']
    y_pred = df['predicted_encoded']

    # Get class-wise metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # Weighted metrics
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    class_names = ["NOT BUG", "BUG"]
    class_metrics = list(zip(class_names, precision, recall, f1, support))
    class_metrics_sorted = sorted(class_metrics, key=lambda x: x[3], reverse=True)

    return {
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "class_metrics": class_metrics_sorted
    }

if __name__ == "__main__":
    result_files = ["colavito-1.csv", "colavito-2.csv", "colavito-3.csv"]

    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0
    accuracy = 0

    bug_prec, bug_rec, bug_f1_score = 0, 0, 0
    not_bug_prec, not_bug_rec, not_bug_f1_score = 0, 0, 0

    for result in result_files:
        df = pd.read_csv(result, usecols=[1, 2, 3], names=['id', 'actual', 'predicted'], header=None)
        
        result = binary_classification(df)
        weighted_precision += result["weighted_precision"]
        weighted_recall += result["weighted_recall"]
        weighted_f1 += result["weighted_f1"]
        accuracy += result["accuracy"]

        for class_name, prec, rec, f1_score, supp in result["class_metrics"]:
            if class_name == "BUG":
                bug_prec += prec
                bug_rec += rec
                bug_f1_score += f1_score
            else:
                not_bug_prec += prec
                not_bug_rec += rec
                not_bug_f1_score += f1_score


    weighted_precision /= len(result_files)
    weighted_recall /= len(result_files)
    weighted_f1 /= len(result_files)
    accuracy /= len(result_files)

    bug_prec /= len(result_files)
    bug_rec /= len(result_files)
    bug_f1_score /= len(result_files)
    not_bug_prec /= len(result_files)
    not_bug_rec /= len(result_files)
    not_bug_f1_score /= len(result_files)

    print("--------------------------------")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(" ")
    print(f"Bug Precision: {bug_prec:.4f}")
    print(f"Bug Recall: {bug_rec:.4f}")
    print(f"Bug F1 Score: {bug_f1_score:.4f}")
    print(" ")
    print(f"Not Bug Precision: {not_bug_prec:.4f}")
    print(f"Not Bug Recall: {not_bug_rec:.4f}")
    print(f"Not Bug F1 Score: {not_bug_f1_score:.4f}")
    print("--------------------------------")