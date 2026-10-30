import json
import csv
from setfit import SetFitModel

def convert_to_label(num):
    label_mapping = {
      0 : "BUG",
      1 : "DOCUMENTATION",
      2 : "FEATURE",
      3 : "QUESTION"
    }
    return label_mapping[num]

if __name__ == "__main__":
    model = SetFitModel.from_pretrained('PeppoCola/FewShotIssueClassifier-NLBSE23')
    dataset = '../dataset/herzig/dataset.json'

    with open(dataset, 'r') as file:
      data = file.read()
    json_data = json.loads(data)

    # Extract issue IDs, descriptions, and labels
    issue_ids = [item['key'] for item in json_data]
    issues = [item['summary'] + '\n' + item['description'] for item in json_data]
    labels = [item['classified'] for item in json_data]

    predictions = model.predict(issues)
    predictions = [pred.item() for pred in predictions]

    # Write results to CSV
    output_file = 'setfit_output.csv'
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for id, actual, pred in zip(issue_ids, labels, predictions):
            writer.writerow([id, actual, convert_to_label(pred)])

    print(f"\nResults written to {output_file}")