import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings; warnings.filterwarnings('ignore')

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
    print()

def result_by_dataset(datsets):
    for ground_truth_file, prediction_file in datasets:
        print(f"*****{ground_truth_file.split('/')[1]}****")
        df_ground_truth = pd.read_csv(ground_truth_file, usecols=[0, 1, 2], names=['project', 'id', 'actual'], header=None)
        df_prediction = pd.read_csv(prediction_file, usecols=[0, 1, 3], names=['project', 'id', 'predicted'], header=None)

        df_ground_truth['project'] = df_ground_truth['project'].str.lower()
        df_prediction['project'] = df_prediction['project'].str.lower()

        df_ground_truth['actual'] = df_ground_truth['actual'].str.upper()

        # INSERT_YOUR_CODE
        # Merge prediction with ground truth using (project, id) as key, left join
        df = pd.merge(
            df_ground_truth,
            df_prediction,
            on=['project', 'id'],
            how='left'
        )

        binary_classification(df)

def combined_result(datasets):
    dfs = []
    
    for ground_truth_file, prediction_file in datasets:
        df_ground_truth = pd.read_csv(ground_truth_file, usecols=[0, 1, 2], names=['project', 'id', 'actual'], header=None)
        df_prediction = pd.read_csv(prediction_file, usecols=[0, 1, 3], names=['project', 'id', 'predicted'], header=None)

        df_ground_truth['project'] = df_ground_truth['project'].str.lower()
        df_prediction['project'] = df_prediction['project'].str.lower()

        df_ground_truth['actual'] = df_ground_truth['actual'].str.upper()

        # Merge prediction with ground truth using (project, id) as key, left join
        df = pd.merge(
            df_ground_truth,
            df_prediction,
            on=['project', 'id'],
            how='left'
        )
        
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    binary_classification(df)
    mismatches = df[df['predicted'] != df['actual']]
    mismatches.to_csv("outputs/combined-mismatches.csv", index=False)


if __name__ == "__main__":
    datasets = [
        ("hand-classified/beetlebox.csv", "outputs/bettlebox-majority.csv"),
        ("hand-classified/bench4bl.csv", "outputs/bench4bl-majority.csv"),
        ("hand-classified/ghrb.csv", "outputs/ghrb-majority.csv"),
        ("hand-classified/yeetal.csv", "outputs/yeetal-majority.csv"),
        ("hand-classified/zhou.csv", "outputs/zhouetal-majority.csv")
    ]

    # result_by_dataset(datasets)
    combined_result(datasets)
    