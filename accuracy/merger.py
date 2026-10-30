import pandas as pd

projects = ["aspectj", "birt", "eclipse", "jdt", "swt", "tomcat"]
techniques = ["buglocator", "bluir", "vsm", "brtracer", "dreamloc", "genloc"]

all_projects_data = []
df_full_bugclassify = pd.read_csv("../bugclassify/outputs/yeetal-majority.csv", names=["Project", "BugId", "Label", "Predicted", "Explanation"], header=None)

for project in projects:
    df_bugclassify = df_full_bugclassify[df_full_bugclassify["Project"] == project]
    df_train_test = pd.read_csv(f"../dataset/train_test_{project}.csv", names=["BugId", "train_test"], skiprows=1)
    print(df_train_test.head())
    merged_df = pd.merge(df_bugclassify, df_train_test, on="BugId", how="inner")
    merged_df['project'] = project  # Add project column
    
    for tech in techniques:
        baseline_file = f"./{tech}/localized_bugs_{project}.csv"
        clean_file = f"./{tech}/localized_bugs_{project}_clean.csv"

        baseline_ids = pd.read_csv(baseline_file)["Accuracy@1"].tolist()
        clean_ids = pd.read_csv(clean_file)["Accuracy@1"].tolist()
        merged_df[f'{tech}_baseline_k@1'] = merged_df['BugId'].isin(baseline_ids)
        merged_df[f'{tech}_clean_k@1'] = merged_df['BugId'].isin(clean_ids)

        baseline_ids = pd.read_csv(baseline_file)["Accuracy@5"].tolist()
        clean_ids = pd.read_csv(clean_file)["Accuracy@5"].tolist()
        merged_df[f'{tech}_baseline_k@5'] = merged_df['BugId'].isin(baseline_ids)
        merged_df[f'{tech}_clean_k@5'] = merged_df['BugId'].isin(clean_ids)

        baseline_ids = pd.read_csv(baseline_file)["Accuracy@10"].tolist()
        clean_ids = pd.read_csv(clean_file)["Accuracy@10"].tolist()
        merged_df[f'{tech}_baseline_k@10'] = merged_df['BugId'].isin(baseline_ids)
        merged_df[f'{tech}_clean_k@10'] = merged_df['BugId'].isin(clean_ids)

    clean_cols = [f"{tech}_clean_k@1" for tech in techniques]
    merged_df.loc[merged_df['Predicted'] != 'BUG', clean_cols] = None

    clean_cols = [f"{tech}_clean_k@5" for tech in techniques]
    merged_df.loc[merged_df['Predicted'] != 'BUG', clean_cols] = None

    clean_cols = [f"{tech}_clean_k@10" for tech in techniques]
    merged_df.loc[merged_df['Predicted'] != 'BUG', clean_cols] = None

    all_projects_data.append(merged_df)

final_df = pd.concat(all_projects_data, ignore_index=True)
final_df.to_csv("./merged_dataset.csv", index=False)
