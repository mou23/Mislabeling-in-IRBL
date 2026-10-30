import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


def accuracy(frame, column):
    if frame.empty:
        return np.nan
    return (frame[column] == True).sum() / len(frame)


def relative_difference(first, second):
    if pd.isna(first) or pd.isna(second):
        return np.nan
    first = float(f"{first * 100:.2f}")
    second = float(f"{second * 100:.2f}")
    if first == 0:
        return np.nan
    return (second - first) / first * 100


def format_percentage(value):
    return "N/A" if pd.isna(value) else f"{value * 100:.2f}%"


def format_difference(value):
    return "N/A" if pd.isna(value) else f"{value:.2f}%"


def calculate_results(frame, techniques):
    bug_only_frame = frame[frame["Predicted"] == "BUG"]
    results = []

    for technique, original_col in techniques.items():
        original_accuracy = accuracy(frame, original_col)
        test_cleaned_accuracy = accuracy(bug_only_frame, original_col)

        results.append({
            "technique": technique,
            "original_accuracy": format_percentage(original_accuracy),
            "test_cleaned_accuracy": format_percentage(test_cleaned_accuracy),
            "difference_original_to_test_cleaned": format_difference(
                relative_difference(original_accuracy, test_cleaned_accuracy)),
        })

    return results


def calculate_fully_cleaned_results(frame, techniques):
    bug_only_frame = frame[frame["Predicted"] == "BUG"]
    results = []

    for technique, (test_cleaned_col, fully_cleaned_col) in techniques.items():
        test_cleaned_accuracy = accuracy(bug_only_frame, test_cleaned_col)
        fully_cleaned_accuracy = accuracy(bug_only_frame, fully_cleaned_col)

        results.append({
            "technique": technique,
            "test_cleaned_accuracy": format_percentage(test_cleaned_accuracy),
            "fully_cleaned_accuracy": format_percentage(fully_cleaned_accuracy),
            "difference_test_cleaned_to_fully_cleaned": format_difference(
                relative_difference(test_cleaned_accuracy, fully_cleaned_accuracy)),
        })

    return results


if __name__ == "__main__":
    dataset = "merged_dataset_genloc_trial_1.csv"
    top_k_values = [1, 5, 10]
    all_results = []

    df = pd.read_csv(dataset)
    test_df = df[df["train_test"] == "test"]

    for k in top_k_values:
        techniques = {
            "vsm": f"vsm_baseline_k@{k}",
            "buglocator": f"buglocator_baseline_k@{k}",
            "bluir": f"bluir_baseline_k@{k}",
            "brtracer": f"brtracer_baseline_k@{k}",
            "dreamloc": f"dreamloc_baseline_k@{k}",
            "genloc": f"genloc_baseline_k@{k}",
        }

        for result in calculate_results(test_df, techniques):
            result["k"] = k
            all_results.append(result)

    results = pd.DataFrame(all_results)
    table_rows = []

    for technique, technique_results in results.groupby("technique", sort=False):
        row = {("Technique", ""): technique.upper()}
        for k in top_k_values:
            k_result = technique_results[technique_results["k"] == k].iloc[0]
            row[(f"Accuracy@{k}", "Original")] = k_result["original_accuracy"]
            row[(f"Accuracy@{k}", "Test-cleaned")] = k_result[
                "test_cleaned_accuracy"
            ]
            row[(f"Accuracy@{k}", "Diff (%)")] = k_result[
                "difference_original_to_test_cleaned"
            ]
        table_rows.append(row)

    table = pd.DataFrame(table_rows)
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    table.to_csv("detailed_accuracy_results.csv", index=False)
    print(table.to_string(index=False))

    fully_cleaned_techniques = {
        "buglocator": (
            "buglocator_baseline_k@{k}",
            "buglocator_clean_k@{k}",
        ),
        "brtracer": (
            "brtracer_baseline_k@{k}",
            "brtracer_clean_k@{k}",
        ),
        "dreamloc": (
            "dreamloc_baseline_k@{k}",
            "dreamloc_clean_k@{k}",
        ),
    }
    fully_cleaned_results = []

    for k in top_k_values:
        techniques = {
            technique: (test_cleaned.format(k=k), fully_cleaned.format(k=k))
            for technique, (test_cleaned, fully_cleaned) in fully_cleaned_techniques.items()
        }
        for result in calculate_fully_cleaned_results(test_df, techniques):
            result["k"] = k
            fully_cleaned_results.append(result)

    fully_cleaned_table_rows = []
    fully_cleaned_results = pd.DataFrame(fully_cleaned_results)

    for technique, technique_results in fully_cleaned_results.groupby("technique", sort=False):
        row = {("Technique", ""): technique.upper()}
        for k in top_k_values:
            k_result = technique_results[technique_results["k"] == k].iloc[0]
            row[(f"Accuracy@{k}", "Test-cleaned")] = k_result[
                "test_cleaned_accuracy"
            ]
            row[(f"Accuracy@{k}", "Fully-cleaned")] = k_result[
                "fully_cleaned_accuracy"
            ]
            row[(f"Accuracy@{k}", "Diff (%)")] = k_result[
                "difference_test_cleaned_to_fully_cleaned"
            ]
        fully_cleaned_table_rows.append(row)

    fully_cleaned_table = pd.DataFrame(fully_cleaned_table_rows)
    fully_cleaned_table.columns = pd.MultiIndex.from_tuples(fully_cleaned_table.columns)
    fully_cleaned_table.to_csv("fully_cleaned_accuracy_results.csv", index=False)
    print("\nTest-cleaned vs. fully-cleaned results")
    print(fully_cleaned_table.to_string(index=False))