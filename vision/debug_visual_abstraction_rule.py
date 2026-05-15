# vision/debug_visual_abstraction_rule.py

import json
import os
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


from reasoning.visual_abstraction_rule_learner import (
    discover_visual_abstraction_rule_for_task,
    print_visual_abstraction_rule,
)


TARGET_TASK_ID = "2d0172a1"


def load_task(task_id):
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


def main():
    task = load_task(TARGET_TASK_ID)

    train_pairs = task.get("train", [])

    print("=" * 60)
    print(f"DEBUG VISUAL ABSTRACTION RULE — TASK {TARGET_TASK_ID}")
    print("=" * 60)

    rule = discover_visual_abstraction_rule_for_task(train_pairs)
    print_visual_abstraction_rule(rule)


if __name__ == "__main__":
    main()