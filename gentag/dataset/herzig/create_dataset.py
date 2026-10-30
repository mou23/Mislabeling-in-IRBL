import os
import json

from setfit import SetFitModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

label_mapping = {
  0 : "bug",
  1 : "documentation",
  2 : "feature",
  3 : "question"
}

label_map = {
  "bug": 0,
  "documentation": 1,
  "feature": 2,
  "question": 3,
  "improvement": 4,
  "": 5,
  "task": 6,
  "rfe": 7,
  "test": 8
}

data_files = os.listdir('data')
total_json = []
for _file in data_files:  
    with open('data/'+_file, 'r') as file:
        data = file.read()

    data_json = json.loads(data)
    total_json.extend(data_json)

with open('dataset.json', "w") as f:
    json.dump(total_json, f, indent=2)