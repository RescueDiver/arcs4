import json
import os

from reasoning.anchor_repair_rule import (
    discover_anchor_repair_rule_for_task,
    apply_anchor_repair_rule,
)


def load_task(task_id):
    base_dir = os.path.dirname(__file__)

    path = os.path.join(
        base_dir,
        "data_failures",
        "extracted_tasks",
        task_id + ".json",
    )

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if "train" in raw:
        return raw

    if task_id in raw:
        return raw[task_id]

    if isinstance(raw, dict) and len(raw) == 1:
        key = next(iter(raw))
        return raw[key]

    raise KeyError(f"Could not find train/test in {path}")


def print_grid(title, grid):
    print()
    print(title)
    print(f"h={len(grid)}, w={len(grid[0]) if grid else 0}")

    for row in grid:
        print(" ".join(str(v) for v in row))


def main():
    task = load_task("20270e3b")
    train_pairs = task["train"]

    rule = discover_anchor_repair_rule_for_task(train_pairs)

    print()
    print("RULE:")
    print(rule)

    pair = train_pairs[0]
    expected = pair["output"]
    predicted = apply_anchor_repair_rule(rule, pair["input"])

    print_grid("EXPECTED PAIR 1", expected)
    print_grid("PREDICTED PAIR 1", predicted)

    print()
    print("DIFF PAIR 1")
    for r in range(len(expected)):
        row = []
        for c in range(len(expected[0])):
            if expected[r][c] == predicted[r][c]:
                row.append(".")
            else:
                row.append("X")
        print(" ".join(row))


if __name__ == "__main__":
    main()