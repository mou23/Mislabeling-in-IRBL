import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import sys

def binary_classification(df):
    print("***************** Binary Classification *****************")

    df['actual'] = np.where(df['actual'] == "BUG", "BUG", "NOT BUG")
    df['predicted'] = np.where(df['predicted'] == "BUG", "BUG", "NOT BUG")
    
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

    print("Class-wise Metrics:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    for class_name, prec, rec, f1_score, supp in class_metrics_sorted:
        print(f"{class_name:<15} {prec:<10.4f} {rec:<10.4f} {f1_score:<10.4f} {supp:<10}")

    print("\nWeighted Metrics:")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    conf_matrix = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = conf_matrix.ravel()

    print("\nConfusion Matrix:")
    print("TP: "+str(TP))
    print("FN: "+str(FN))
    print("FP: "+str(FP))
    print("TN: "+str(TN))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python result_process_setfit.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = pd.read_csv(file_path, usecols=[0, 1, 2], names=['id', 'actual', 'predicted'], header=None)

    binary_classification(df)