import json
import xml.etree.ElementTree as ET

from openai import OpenAI
import csv

import concurrent.futures
import threading
from pydantic import BaseModel
from typing import Generator, List, Optional
import re, html
import os, sys
import argparse
from lxml import etree

file_lock = threading.Lock()

output_file = None

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

class Issue(BaseModel):
    project: Optional[str] = None
    id: str
    title: str
    description: Optional[str] = None
    label: Optional[str] = None
    discussion: Optional[List[str]] = None

class HerzigDataset:
    def __init__(self, file_path="dataset/herzig/dataset_full.json") -> None:
        with open(file_path, 'r') as file:
            data = file.read()
        self.json_data = json.loads(data)

    def get_data(self) -> Generator[Issue, None, None]:
        for row in self.json_data:
            issue = Issue(id=row['key'], title=row['summary'], description=row['description'], label=row['classified'])
            if "comments" in row:
                issue.discussion = [comment for comment in row["comments"]]
            yield issue

class YeEtal:
    def __init__(self, folder_path="dataset/yeetal/") -> None:
        self.folder_path = folder_path

    def _clean_text(self, text: str) -> str:
        if text is None:
            return ""
        # Replace Unicode escape sequences with actual characters
        text = re.sub(r'\\u([0-9A-Fa-f]{4})', lambda m: chr(int(m.group(1), 16)), text)
        # Unescape HTML entities (if present)
        text = html.unescape(text)
        # Strip leading/trailing spaces
        return text.strip()

    def get_data(self) -> Generator[Issue, None, None]:
        for xml_file in [f for f in os.listdir(self.folder_path) if f.endswith('.xml')]:
            project = xml_file.split("-")[0]
            with open(os.path.join(self.folder_path, xml_file), 'r', encoding="utf-8", errors='replace') as file:
                data = file.read()
            xml_data = ET.fromstring(data)

            for table in xml_data.findall(".//table"):
                bug_id = table.find(".//column[@name='bug_id']")
                summary = table.find(".//column[@name='summary']")
                description = table.find(".//column[@name='description']")

                bug_id = self._clean_text(bug_id.text if bug_id is not None else "")
                summary = self._clean_text(summary.text if summary is not None else "")
                description = self._clean_text(description.text if description is not None else "")

                print("[*] ", bug_id) #, summary, description)
                issue = Issue(project=project, id=str(bug_id), title=summary, description=description)
                yield issue

class bench4BL:
    def __init__(self, folder_path="dataset/bench4bl/") -> None:
        self.folder_path = folder_path

    def _clean_text(self, text: str) -> str:
        if text is None:
            return ""
        # Replace Unicode escape sequences with actual characters
        text = re.sub(r'\\u([0-9A-Fa-f]{4})', lambda m: chr(int(m.group(1), 16)), text)
        # Unescape HTML entities (if present)
        text = html.unescape(text)
        # Strip leading/trailing spaces
        return text.strip()

    def get_data(self) -> Generator[Issue, None, None]:
        for xml_file in [f for f in os.listdir(self.folder_path) if f.endswith('.xml')]:
            project = xml_file.split(".")[0]
            with open(os.path.join(self.folder_path, xml_file), 'r', encoding="utf-8", errors='replace') as file:
                data = file.read()
            xml_data = ET.fromstring(data)

            for bug in xml_data.findall(".//bug"):
                bug_id = bug.get("id")
                info = bug.find("buginformation")
                if info is not None:
                    summary = self._clean_text(info.findtext("summary", ""))
                    description = self._clean_text(info.findtext("description", ""))
                else:
                    summary = description = ""

                bug_id = self._clean_text(bug_id)
                print("[*]", project, bug_id)
                yield Issue(project=project, id=bug_id, title=summary, description=description)

class Zhouetal(bench4BL):
    def __init__(self, folder_path="dataset/Buglocator/data/") -> None:
        self.folder_path = folder_path

class GhrbDataset(YeEtal):
    def __init__(self, folder_path="dataset/ghrbdataset/") -> None:
        self.folder_path = folder_path

    def get_data(self) -> Generator[Issue, None, None]:
        for xml_file in [f for f in os.listdir(self.folder_path) if f.endswith('.xml')]:
            # Extract project name from filename (e.g., "jsoup.xml" -> "jsoup")
            project = xml_file.replace(".xml", "")
            with open(os.path.join(self.folder_path, xml_file), 'r', encoding="utf-8", errors='replace') as file:
                data = file.read()
            xml_data = ET.fromstring(data)

            for table in xml_data.findall(".//table"):
                bug_id_elem = table.find(".//column[@name='bug_id']")
                summary_elem = table.find(".//column[@name='summary']")
                description_elem = table.find(".//column[@name='description']")

                bug_id = self._clean_text(bug_id_elem.text if bug_id_elem is not None else "")
                summary = self._clean_text(summary_elem.text if summary_elem is not None else "")
                description = self._clean_text(description_elem.text if description_elem is not None else "")

                issue = Issue(project=project, id=str(bug_id), title=summary, description=description)
                yield issue

class BettleboxDataset(YeEtal):
    def __init__(self, folder_path="./dataset/bettlebox/") -> None:
        self.folder_path = folder_path

    def clean_xml_bytes(self, raw: bytes) -> bytes:
        _ILLEGAL_XML_RE = re.compile(
            r'[\x00-\x08\x0B\x0C\x0E-\x1F]'
        )
        """Remove all illegal XML 1.0 control characters from bytes."""
        # decode first (replace errors)
        text = raw.decode("utf-8", errors="replace")
        # remove illegal chars
        text = _ILLEGAL_XML_RE.sub("", text)
        # encode back to bytes for lxml
        return text.encode("utf-8")

    def get_data(self) -> Generator[Issue, None, None]:
        for xml_file in [f for f in os.listdir(self.folder_path) if f.endswith('.xml')]:
            with open(os.path.join(self.folder_path, xml_file), "rb") as f:
                raw = f.read()
            raw = self.clean_xml_bytes(raw)
            # lxml parser with recover=True handles illegal characters and malformed XML
            parser = etree.XMLParser(recover=True, encoding="utf-8")
            root = etree.fromstring(raw, parser=parser)

            # Bettlebox dataset structure
            for bug in root.findall(".//bug"):
                bug_id_elem = bug.find("bug_id")
                summary_elem = bug.find("summary")
                description_elem = bug.find("description")

                bug_id = self._clean_text(bug_id_elem.text if bug_id_elem is not None else "")
                summary = self._clean_text(summary_elem.text if summary_elem is not None else "")
                description = self._clean_text(description_elem.text if description_elem is not None else "")

                # Extract project from bug_id
                project = bug_id.split("/")[0] if "/" in bug_id else bug_id

                issue = Issue(
                    project=project,
                    id=str(bug_id),
                    title=summary,
                    description=description
                )

                yield issue


def bug_predict(issue: Issue, explanation=True):
    role_system = '''
As an expert software maintainer, your goal is to accurately classify software issues into one of the following predefined labels based on the provided title and description. The definitions of the labels are as follows:

* BUG 
    - the issue reports a runtime error (Null Pointer Exception, crashes)
    - the code has to be changed semantically to fixing something (including tests) that is broken or not working properly
    - it ﬁxes runtime or memory issues caused by defects
    - it reports inconsistent functionality (handling of threads, state inconsistencies, non-threadsafe operations)
* BUILD_SYSTEM
    - issues related to the build configurations such as error in config files (pom.xml, build.xml)
    - compilation errors due to missing dependencies (jar, dll, libraries)
* DESIGN_DEFECT
    - issues related to poor class design
* DOCUMENTATION 
    - the issue is related to the documentation/website of the software
* IMPROVEMENT
    - requires enhancements/optimizations to existing features, suggests calling a different function
    - suggests improving error messages, error handling, memory management
    - suggests throwing or changing exception messages, improve wrong exceptions returned, adding assertions or logging
    - suggests improving performance or adaptibility to different versions or platforms
* REFACTORING
    - it requests to move code into other packages, classes, or methods
    - it requests to rename variables, methods, classes, packages, or configuration options
* RFE
    - issue asks for new features/requirements or customizing existing features
* TEST
    - issues related to test coverage, testing frameworks, test case initialization
    - identifies typos/improvements in test methods or test logic
* BACKPORT
* CLEANUP
* SPEC
    - changes to specification (rather than documentation or code)
* TASK
* OTHER
'''

    role_user = '''
Instructions:
* Carefully analyze the title and description of the issue.
* Assign the most appropriate label from the list above based on the nature of the issue.
{explanation}

Title: {title}

Description: {description}
'''
    # comments = issue['comments']
    # diffs = issue['diffs']

    explanation_string = '* Provide a brief explanation of your decision.' if explanation else ''
    role_user = role_user.format(title=issue.title, description=issue.description, explanation=explanation_string)

    # if comments:
    #     role_user += f"\nComments: {comments[:20]}"

    # if diffs:
    #     role_user += f"\nDiffs: \n"
    #     short_diffs = [diff["diff"] for diff in diffs if diff["diff"].count('\n') < 70][:10]
    #     for diff in short_diffs:
    #         role_user += f"File: {diff['file']}\n"
    #         role_user += f"{diff['diff']}\n\n"

    messages = [
        {"role": "system", "content": role_system},
        {"role": "user", "content": role_user}
    ]

    class ResponseModel(BaseModel):
        label: str

    class ResponseModelWithExplanation(BaseModel):
        label: str
        explanation: str

    response_format = ResponseModelWithExplanation if explanation else ResponseModel

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=response_format,
            temperature=0
            # max_tokens=2048
        )
        
        response = response.choices[0].message.content
        if response.startswith("```json"):
            response = response[7:-3]

        j = json.loads(response)
        prediction = j['label']
        explanation = j['explanation'] if explanation else None

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
            
DATASETS = {
    "herzig": HerzigDataset,
    "yeetal": YeEtal,
    "bench4bl": bench4BL,
    "zhou": Zhouetal,
    "ghrb": GhrbDataset,
    "bettlebox": BettleboxDataset,
}

def main(dataset: str, mode: str):
    data = DATASETS[dataset]()

    if mode == "sequential":
        for issue in data.get_data():
            try:
                bug_predict(issue)
            except Exception:
                print("Error printing issue", issue.id)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        futures = [executor.submit(bug_predict, issue) for issue in data.get_data()]
        for future in concurrent.futures.as_completed(futures):
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file")
    parser.add_argument("dataset", choices=DATASETS.keys())
    parser.add_argument("mode", choices=["sequential", "parallel"])
    args = parser.parse_args()

    output_file = args.output_file
    main(args.dataset, args.mode)
