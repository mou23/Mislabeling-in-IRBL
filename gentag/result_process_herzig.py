import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import sys

def multi_label_classification(df):
    print("***************** Multi Label Classification *****************")

    all_labels = ["BUG", "BUILD_SYSTEM", "IMPROVEMENT", "DESIGN_DEFECT", "DOCUMENTATION", "REFACTORING", "RFE", "TEST", "BACKPORT", "CLEANUP", "SPEC", "TASK", "UNKNOWN", "OTHER"]

    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    df['actual_encoded'] = label_encoder.transform(df['actual'])
    df['predicted_encoded'] = label_encoder.transform(df['predicted'].str.upper())

    y_true = df['actual_encoded']
    y_pred = df['predicted_encoded']

    # Get class-wise metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(all_labels)), average=None, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(all_labels)))

    # Extract TP, FP, FN, TN for each class
    tp = cm.diagonal()
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    # tn = cm.sum() - (fp + fn + tp)

    # Weighted metrics
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    class_names = label_encoder.classes_
    class_metrics = list(zip(class_names, precision, recall, f1, support, tp, fn, fp))
    class_metrics_sorted = sorted(class_metrics, key=lambda x: x[3], reverse=True)

    print("Class-wise Metrics:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10} {'TP':<10} {'FN':<10} {'FP':<10}")
    for class_name, prec, rec, f1_score, supp, tp, fn, fp in class_metrics_sorted:
        print(f"{class_name:<15} {prec:<10.4f} {rec:<10.4f} {f1_score:<10.4f} {supp:<10} {tp:<10} {fn:<10} {fp:<10}")

    # Print overall weighted metrics
    print("\nWeighted Metrics:")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # conf_matrix = confusion_matrix(y_true, y_pred)
    # print("\nConfusion Matrix:")
    # print(" " * 10 + " ".join([f"{class_name:<5}" for class_name in class_names]))
    # for i, row in enumerate(conf_matrix):
    #     print(f"{class_names[i]:<10} " + " ".join([f"{value:<10}" for value in row]))

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
        print("Usage: python result_process_herzig.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = pd.read_csv(file_path, usecols=[1, 2, 3], names=['id', 'actual', 'predicted'], header=None)

    # multi_label_classification(df)
    binary_classification(df)