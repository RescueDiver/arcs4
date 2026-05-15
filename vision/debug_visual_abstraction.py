# vision/debug_visual_abstraction.py

import json
import os
import sys


# ============================================================
# PROJECT PATH SETUP
# ============================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


from reasoning.visual_abstraction_discovery import (
    discover_visual_abstractions,
    print_visual_abstractions,
)


# ============================================================
# TASK TO DEBUG
# ============================================================

TARGET_TASK_ID = "2d0172a1"


# ============================================================
# TASK LOADER
# copied from run_oneV2.py style
# ============================================================

def load_task(task_id):
    """
    Load a single ARC task by ID.

    Searches the same common locations as run_oneV2.py.
    """
    exact_filename = f"{task_id}.json"

    short_id = task_id
    if len(task_id) > 3 and task_id[:3].isdigit():
        short_id = task_id[:3]

    candidate_paths = [
        os.path.join(BASE_DIR, "data_failures", "extracted_tasks", exact_filename),
        os.path.join(BASE_DIR, "data_failures", exact_filename),
        os.path.join(BASE_DIR, "data", exact_filename),
        os.path.join(BASE_DIR, "data", f"{short_id}.json"),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            print(f"Loaded task from: {path}")

            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            if isinstance(raw, dict) and task_id in raw:
                return raw[task_id]

            if isinstance(raw, dict) and ("train" in raw or "test" in raw):
                return raw

            if isinstance(raw, dict) and len(raw) == 1:
                only_value = next(iter(raw.values()))

                if isinstance(only_value, dict) and (
                    "train" in only_value or "test" in only_value
                ):
                    return only_value

            raise ValueError(f"Unexpected task format in {path}")

    raise FileNotFoundError(
        "Task file not found in any expected location:\n"
        + "\n".join(candidate_paths)
    )


# ============================================================
# MAIN
# ============================================================

def main():
    task = load_task(TARGET_TASK_ID)

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    print("=" * 60)
    print(f"DEBUG VISUAL ABSTRACTION — TASK {TARGET_TASK_ID}")
    print("=" * 60)
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Test pairs : {len(test_pairs)}")

    for idx, pair in enumerate(train_pairs, start=1):
        print("\n" + "=" * 60)
        print(f"TRAIN PAIR {idx}")
        print("=" * 60)

        summary = discover_visual_abstractions(pair["input"])
        print_visual_abstractions(summary)

    for idx, pair in enumerate(test_pairs, start=1):
        print("\n" + "=" * 60)
        print(f"TEST PAIR {idx}")
        print("=" * 60)

        summary = discover_visual_abstractions(pair["input"])
        print_visual_abstractions(summary)


if __name__ == "__main__":
    main()