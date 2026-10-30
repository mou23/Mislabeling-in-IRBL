import requests
import json

JIRA_URL = 'https://issues.apache.org/jira/rest/api/2/issue/'


with open('dataset.json', 'r') as file:
    data = file.read()
json_data = json.loads(data)

for issue in json_data:
    print("[*] Doing: ", issue['key'])
    id = issue['key']
    r = requests.get(JIRA_URL + id)
    issue_data = r.json()

    comments = [item["body"] for item in issue_data["fields"]["comment"]["comments"]]

    print(comments)
    with open(f'comments/{id}.json', 'w') as file:
        json.dump(issue_data, file, indent=2)