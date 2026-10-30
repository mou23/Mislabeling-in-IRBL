import pandas as pd
import os

def remove_duplicates_from_csv(file_path):
    """
    Args:
        file_path: Path to the CSV file
    """
    print(f"Processing {file_path}...")

    df = pd.read_csv(file_path, header=None)
    initial_count = len(df)

    # Drop duplicates based on column 1 (ID column)
    # CSV structure: project, id, label, prediction, explanation
    df_cleaned = df.drop_duplicates(subset=[1], keep='first')

    final_count = len(df_cleaned)
    duplicates_removed = initial_count - final_count

    df_cleaned.to_csv(file_path, index=False, header=False)

    print(f"  Initial rows: {initial_count}")
    print(f"  Final rows: {final_count}")
    print(f"  Duplicates removed: {duplicates_removed}")
    print()

def main():
    bettlebox_files = [
        'bettlebox-1.csv',
        'bettlebox-2.csv',
        'bettlebox-3.csv',
    ]

    for filename in bettlebox_files:
        file_path = os.path.join('.', filename)

        if os.path.exists(file_path):
            remove_duplicates_from_csv(file_path)
        else:
            print(f"Warning: {file_path} not found, skipping...")

if __name__ == "__main__":
    main()
