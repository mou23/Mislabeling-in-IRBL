import pandas as pd
import xml.etree.ElementTree as ET
import sys

project_name = sys.argv[1]
prediction_file = sys.argv[2]

df = pd.read_csv(prediction_file, header=None)
df.columns = ["bug_id", "actual", "predicted"]

bug_ids_to_keep = df[df["predicted"] == "BUG"]["bug_id"]
print(set(bug_ids_to_keep))

xml_file = f"{project_name}-filtered.xml"
tree = ET.parse(xml_file)
root = tree.getroot()

for database in root.findall("database"):
    for table in database.findall("table"):
        bug_id_element = table.find("./column[@name='bug_id']")
        if bug_id_element is not None:
            bug_id = bug_id_element.text
            if bug_id not in bug_ids_to_keep:
                print(bug_id)
                database.remove(table)

output_file = f"{project_name}-bugs-only.xml"
tree = ET.ElementTree(root)
tree.write(output_file, encoding="utf-8", xml_declaration=True)
print(f"Filtered XML saved to {output_file}")