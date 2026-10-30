import json

with open('dataset_full.json', 'r') as file:
    data = file.read()
    json_data = json.loads(data)

comment_length = 0
comment_count = 0
comment_length_highest = 0

diff_length = 0
diff_count = 0
diff_length_highest = 0

for item in json_data:
    for comment in item['comments']:
        comment_length += comment.count('\n')
        comment_count += 1
        if comment.count('\n') > comment_length_highest:
            comment_length_highest = comment.count('\n')

    for diff in item['diffs']:
        diff_length += diff['diff'].count('\n')
        diff_count += 1
        if diff['diff'].count('\n') > diff_length_highest:
            diff_length_highest = diff['diff'].count('\n')

print(f"Average comment length: {comment_length/comment_count}")
print(f"Highest comment length: {comment_length_highest}")

print(f"Average diff length: {diff_length/diff_count}")
print(f"Highest diff length: {diff_length_highest}")
