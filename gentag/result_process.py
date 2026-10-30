import pandas as pd
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python result_process.py <csv_file>")
        exit()

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path, header=None)

    df.columns = ['Project', 'BugID', 'Label', 'Predicted', 'Explanation']
        
    total_entries = len(df)
    bug_entries = len(df[df['Predicted'] == 'BUG'])
    
    print(f"Bugs: {bug_entries}/{total_entries} ({bug_entries/total_entries*100:.2f}%)")
    print(f"Non-bugs: {total_entries - bug_entries}/{total_entries} ({(total_entries - bug_entries)/total_entries*100:.2f}%)")

    unique_projects = df['Project'].nunique()
    print(f"\nTotal unique projects: {unique_projects}")
    
    print(f"Project Distribution:")
    project_counts = df['Project'].value_counts()
    for project, count in project_counts.items():
        print(f"   {project}: {count} entries")
    
    # Show label distribution
    print(f"\nLabel Distribution:")
    label_counts = df['Predicted'].value_counts()
    for label, count in label_counts.items():
        print(f"   {label}: {count} entries ({(count/total_entries)*100:.2f}%)")