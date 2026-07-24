# quarantine_motif_layout_full_wins.py
import json
import os
import shutil


DATA_PATH = os.path.join("data", "data.json")
FULL_IDS_PATH = os.path.join(
    "data_failures",
    "motif_layout_audit",
    "fully_solved_by_motif_layout.txt",
)
OUT_DIR = os.path.join(
    "data_failures",
    "extracted_tasks",
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_ids(path):
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def main():
    data = load_json(DATA_PATH)
    task_ids = load_ids(FULL_IDS_PATH)

    print()
    print("=" * 80)
    print("QUARANTINE MOTIF_LAYOUT_RULE FULL WINS")
    print("=" * 80)

    if not task_ids:
        print("No fully solved motif_layout_rule task IDs found.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    copied = []

    for task_id in task_ids:
        if task_id not in data:
            print(f"[MISSING] {task_id}")
            continue

        out_path = os.path.join(OUT_DIR, f"{task_id}.json")
        save_json(out_path, data[task_id])
        copied.append(task_id)

        print(f"[COPIED] {task_id} -> {out_path}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Copied motif full wins back to failures: {len(copied)}")
    print("IDs:")
    for task_id in copied:
        print(f"  {task_id}")


if __name__ == "__main__":
    main()