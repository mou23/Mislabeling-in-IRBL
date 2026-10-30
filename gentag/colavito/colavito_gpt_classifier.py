import json
import xml.etree.ElementTree as ET

from openai import OpenAI
import pandas as pd
import csv

import concurrent.futures
import threading
from pydantic import BaseModel
from typing import Generator, List, Optional
import re, html
import os, sys

file_lock = threading.Lock()

output_file = sys.argv[1]

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

class Issue(BaseModel):
    project: Optional[str] = None
    id: str
    title: str
    description: Optional[str] = None
    label: Optional[str] = None

class HerzigDataset:
    def __init__(self, file_path="../dataset/herzig/dataset_full.json") -> None:
        with open(file_path, 'r') as file:
            data = file.read()
        self.json_data = json.loads(data)

    def get_data(self) -> Generator[Issue, None, None]:
        for row in self.json_data:
            issue = Issue(id=row['key'], title=row['summary'], description=row['description'], label=row['classified'])
            yield issue


            
def bug_predict(issue: Issue, explanation=True):

    role_user = '''
You are provided with an issue in this format:
Title : """the title of the issue"""
Body : """the body of the issue"""
You have to assign it a label.

Possible labels are: {label_list}

The "bug" label is used to identify an issue report that describes a problem or error within the
software or codebase. It indicates that something is not functioning as intended or producing unexpected
results. Bug reports help developers identify and fix issues to improve the overall quality and
reliability of the software.

The "feature" label is applied to an issue report that suggests the addition or enhancement of a new
capability or functionality within the software. It indicates a request for the implementation of a
specific feature that does not currently exist. Feature requests help the development team understand
the needs and requirements of the users and can influence the future development roadmap.

The "question" label is used when a user or contributor has a query or needs clarification regarding
the software or codebase. It indicates that the issue report is seeking information, guidance, or
assistance. Questions can be related to functionality, usage, best practices, or any other topic that
requires clarification.

 The "documentation" label is applied to an issue report that suggests improvements, updates, or
corrections to the software's documentation. It indicates that the issue is focused on the written
instructions, guides, or explanations provided alongside the codebase. Documentation issues help ensure
that the information available to users and contributors is accurate, comprehensive, and up-to-date,
thus facilitating better understanding and adoption of the software.

 The "discard" label is used when you can't decide due to lack of information contained in the issue.

Title: {title}
Body: {description}
'''
    label_list = ["bug", "feature", "question", "documentation", "discard"]
    role_user = role_user.format(title=issue.title, description=issue.description, label_list=label_list)

    messages = [
        {"role": "user", "content": role_user}
    ]

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format = { 
                "type": "json_schema",
                "json_schema": {
                    "name": "output_format",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reasoning": {
                                "type": "string",
                                "description": "The step by step reasoning to assign the correct label to the issue."
                            },
                            "label": {
                                "type": "string",
                                "description": f"The label to assign to the issue. Possible labels are: {label_list}"
                            },
                        },
                        "required": ["reasoning", "label"],
                        "additionalProperties": False
                    }
                }
            },
            temperature=0
        )
        
        response = response.choices[0].message.content
        if response.startswith("```json"):
            response = response[7:-3]

        j = json.loads(response)
        prediction = j['label']
        explanation = j['reasoning']

        with file_lock:
            with open(output_file, 'a', encoding='utf-8', errors='ignore', newline="") as csv_file:
                # print(f"[*] Issue: {issue.id}", response)
                writer = csv.writer(csv_file)
                if explanation:
                    writer.writerow([issue.project, issue.id, issue.label, prediction, explanation])
                    # writer.writerow([issue.id, issue.label, prediction, explanation])
                else:
                    writer.writerow([issue.id, issue.label, prediction])
    except Exception as e:
        print(f"[!] Error doing {issue.id}: {e}")
            
def main():
    data = HerzigDataset()

    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        futures = []
        for issue in data.get_data():
            futures.append(executor.submit(bug_predict, issue))
        
        for future in concurrent.futures.as_completed(futures):
            future.result()  

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python colavito_gpt_classifier.py <output_file>")
        sys.exit(1)

    main()
