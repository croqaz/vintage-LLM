import json

sessions = []

with open('exam_sessions.json', 'r') as f:
    sessions.extend(json.load(f))

with open('exam_sessions2.json', 'r') as f:
    sessions.extend(json.load(f))

with open('exam_sessions_combined.jsonl', 'w') as f:
    print(f"Total sessions combined: {len(sessions)}")
    for session in sessions:
        f.write(json.dumps(session) + '\n')
