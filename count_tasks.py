import json
import os

base_dir = os.path.dirname(__file__)
path = os.path.join(base_dir, "data", "data.json")

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

task_ids = [
    k for k, v in data.items()
    if isinstance(v, dict) and ("train" in v or "test" in v)
]

print("Task count:", len(task_ids))
print()
print("First 10 task IDs:")
for task_id in task_ids[:10]:
    print(task_id)