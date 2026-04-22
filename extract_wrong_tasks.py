import json
import os


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_wrong_task_ids(path):
    task_ids = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue
            if line.startswith("TASKS WITH AT LEAST ONE WRONG TRAIN PAIR"):
                continue
            if line.startswith("="):
                continue

            task_ids.append(line)

    return task_ids


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def extract_wrong_tasks():
    base_dir = os.path.dirname(__file__)

    data_path = os.path.join(base_dir, "data", "data.json")
    wrong_tasks_path = os.path.join(base_dir, "data_failures", "wrong_tasks_only.txt")
    output_dir = os.path.join(base_dir, "data_failures", "extracted_tasks")

    ensure_dir(output_dir)

    all_tasks = load_json(data_path)
    wrong_task_ids = load_wrong_task_ids(wrong_tasks_path)

    found = 0
    missing = []

    for task_id in wrong_task_ids:
        if task_id not in all_tasks:
            missing.append(task_id)
            continue

        task_data = {task_id: all_tasks[task_id]}
        output_path = os.path.join(output_dir, f"{task_id}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(task_data, f, indent=2)

        found += 1

    print("=" * 60)
    print("EXTRACT WRONG TASKS SUMMARY")
    print("=" * 60)
    print(f"Requested wrong tasks: {len(wrong_task_ids)}")
    print(f"Extracted task files : {found}")
    print(f"Missing task ids     : {len(missing)}")

    if missing:
        print("\nMissing IDs:")
        for task_id in missing:
            print(f"  {task_id}")

    print(f"\nOutput folder: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    extract_wrong_tasks()