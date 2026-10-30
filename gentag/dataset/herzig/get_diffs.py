from pydriller import Repository
import re
import json
from collections import defaultdict

with open('dataset.json', 'r') as file:
    data = file.read()
    json_data = json.loads(data)

diff_map = defaultdict(list)

def get_ticket_diffs(repo_path, tickets):
    ticket_patterns = {ticket: re.compile(rf"\b{ticket}\b", re.IGNORECASE) for ticket in tickets}
    for commit in Repository(repo_path).traverse_commits():
        print("[*] Doing: ", commit.hash[:10])
        for ticket, pattern in ticket_patterns.items():
            if pattern.search(commit.msg):  # If ticket ID is found in commit message
                print(f"\nTicket: {ticket} | Commit: {commit.hash}\nMessage: {commit.msg}\n")
                
                for modified_file in commit.modified_files:
                    diff_map[ticket].append({
                        "commit": commit.hash,
                        "file": modified_file.filename,
                        "diff": modified_file.diff
                    })

if __name__ == "__main__":
    tickets = [issue['key'] for issue in json_data]
    tickets_http = [ticket for ticket in tickets if ticket.startswith("HTTP")]
    tickets_jcr = [ticket for ticket in tickets if ticket.startswith("JCR")]
    tickets_lucene = [ticket for ticket in tickets if ticket.startswith("LUCENE")]

    # clone the repos
    get_ticket_diffs("./httpcomponents-client/", tickets_http)
    get_ticket_diffs("./jackrabbit/", tickets_jcr)
    get_ticket_diffs("./lucene/", tickets_lucene)

    for i, item in enumerate(json_data):
        json_data[i]['diffs'] = diff_map[item['key']]
        with open(f"comments/{item['key']}.json", 'r') as file:
            data = file.read()
            data = json.loads(data)
            json_data[i]['comments'] = [comment["body"] for comment in data["fields"]["comment"]["comments"]]

    with open('dataset_full.json', 'w') as file:
        json.dump(json_data, file, indent=2)