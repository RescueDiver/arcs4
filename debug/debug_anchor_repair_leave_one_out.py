import json
import os
import tkinter as tk

from reasoning.anchor_repair_rule import (
    discover_anchor_repair_rule_for_task,
    apply_anchor_repair_rule,
)


CELL = 28

COLORS = {
    0: "#000000",
    1: "#0074D9",
    2: "#FF4136",
    3: "#2ECC40",
    4: "#FFDC00",
    5: "#AAAAAA",
    6: "#F012BE",
    7: "#FF851B",
    8: "#7FDBFF",
    9: "#870C25",
}


def grid_shape(grid):
    if grid is None:
        return "None"

    h = len(grid)
    w = len(grid[0]) if h else 0
    return f"{h}x{w}"


def load_task(task_id):
    base_dir = os.path.dirname(__file__)

    data_path = os.path.join(base_dir, "data", "data.json")
    failure_path = os.path.join(
        base_dir,
        "data_failures",
        "extracted_tasks",
        task_id + ".json",
    )

    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if task_id in data:
            return data[task_id]

    if os.path.exists(failure_path):
        with open(failure_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if "train" in raw:
            return raw

        if task_id in raw:
            return raw[task_id]

        if isinstance(raw, dict) and len(raw) == 1:
            key = next(iter(raw))
            return raw[key]

    raise FileNotFoundError(task_id)


def draw_grid(parent, title, grid):
    box = tk.Frame(parent)
    box.pack(side=tk.LEFT, padx=10, pady=8, anchor="n")

    tk.Label(
        box,
        text=f"{title}  {grid_shape(grid)}",
        font=("Arial", 11, "bold"),
    ).pack()

    if grid is None:
        tk.Label(
            box,
            text="NONE",
            font=("Arial", 18, "bold"),
            fg="red",
            width=12,
            height=4,
            relief=tk.SOLID,
        ).pack()
        return

    h = len(grid)
    w = len(grid[0]) if h else 0

    canvas = tk.Canvas(
        box,
        width=w * CELL,
        height=h * CELL,
        bg="white",
        highlightthickness=1,
        highlightbackground="black",
    )
    canvas.pack()

    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            x1 = c * CELL
            y1 = r * CELL
            x2 = x1 + CELL
            y2 = y1 + CELL

            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=COLORS.get(value, "#FFFFFF"),
                outline="#555555",
            )


def summarize_rule(rule):
    if rule is None:
        return "RULE: None"

    return (
        f"RULE: {rule.get('rule_type')} | "
        f"anchor={rule.get('anchor_color')} | "
        f"structure={rule.get('structure_color')} | "
        f"background={rule.get('background_color')} | "
        f"exact={rule.get('exact_count')}/{rule.get('pair_count')} | "
        f"base={rule.get('base_selector')} | "
        f"base_ref={rule.get('base_anchor_ref')} | "
        f"patch_ref={rule.get('patch_anchor_ref')} | "
        f"shift={rule.get('shift_name')} | "
        f"canvas={rule.get('canvas_mode')}"
    )


def add_section(parent, text):
    tk.Label(
        parent,
        text=text,
        font=("Arial", 15, "bold"),
        anchor="w",
    ).pack(fill=tk.X, padx=10, pady=(18, 2))


def main():
    task_id = input("Task id: ").strip()
    task = load_task(task_id)

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    print()
    print("=" * 80)
    print(f"ANCHOR REPAIR LEAVE-ONE-OUT: {task_id}")
    print("=" * 80)

    loo_results = []

    for hidden_index in range(len(train_pairs)):
        learning_pairs = [
            pair
            for i, pair in enumerate(train_pairs)
            if i != hidden_index
        ]

        hidden_pair = train_pairs[hidden_index]

        rule = discover_anchor_repair_rule_for_task(
            learning_pairs,
        )

        predicted = apply_anchor_repair_rule(
            rule,
            hidden_pair["input"],
        )

        exact = predicted == hidden_pair["output"]

        print()
        print(f"HIDDEN TRAIN PAIR {hidden_index + 1}")
        print(summarize_rule(rule))
        print(f"exact: {exact}")
        print(f"expected shape : {grid_shape(hidden_pair['output'])}")
        print(f"predicted shape: {grid_shape(predicted)}")

        loo_results.append({
            "hidden_index": hidden_index,
            "rule": rule,
            "input": hidden_pair["input"],
            "expected": hidden_pair["output"],
            "predicted": predicted,
            "exact": exact,
        })

    full_rule = discover_anchor_repair_rule_for_task(
        train_pairs,
    )

    test_results = []

    print()
    print("=" * 80)
    print("FULL TRAIN TEST PREDICTIONS")
    print("=" * 80)
    print(summarize_rule(full_rule))

    for test_index, pair in enumerate(test_pairs):
        predicted = apply_anchor_repair_rule(
            full_rule,
            pair["input"],
        )

        print()
        print(f"TEST PAIR {test_index}")
        print(f"prediction shape: {grid_shape(predicted)}")

        test_results.append({
            "test_index": test_index,
            "input": pair["input"],
            "predicted": predicted,
        })

    root = tk.Tk()
    root.title(f"ANCHOR REPAIR LOO - {task_id}")

    tk.Label(
        root,
        text=f"ANCHOR REPAIR LEAVE-ONE-OUT: {task_id}",
        font=("Arial", 16, "bold"),
    ).pack(pady=8)

    outer = tk.Frame(root)
    outer.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(outer)
    y_scroll = tk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
    x_scroll = tk.Scrollbar(outer, orient=tk.HORIZONTAL, command=canvas.xview)

    scroll_frame = tk.Frame(canvas)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(
        yscrollcommand=y_scroll.set,
        xscrollcommand=x_scroll.set,
    )

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    x_scroll.pack(side=tk.BOTTOM, fill=tk.X)

    add_section(scroll_frame, "HONEST HIDDEN TRAIN PAIRS")

    for item in loo_results:
        hidden_number = item["hidden_index"] + 1

        tk.Label(
            scroll_frame,
            text=(
                f"HIDDEN TRAIN PAIR {hidden_number} | "
                f"exact={item['exact']} | "
                f"{summarize_rule(item['rule'])}"
            ),
            font=("Arial", 12, "bold"),
            anchor="w",
            wraplength=1600,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=10, pady=(12, 0))

        row = tk.Frame(scroll_frame)
        row.pack(anchor="w")

        draw_grid(row, "HIDDEN INPUT", item["input"])
        draw_grid(row, "EXPECTED", item["expected"])
        draw_grid(row, "PREDICTED", item["predicted"])

    add_section(scroll_frame, "FULL-TRAIN TEST PREDICTION")

    tk.Label(
        scroll_frame,
        text=summarize_rule(full_rule),
        font=("Arial", 12, "bold"),
        anchor="w",
        wraplength=1600,
        justify=tk.LEFT,
    ).pack(fill=tk.X, padx=10, pady=(12, 0))

    for item in test_results:
        tk.Label(
            scroll_frame,
            text=f"TEST PAIR {item['test_index']}",
            font=("Arial", 13, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(12, 0))

        row = tk.Frame(scroll_frame)
        row.pack(anchor="w")

        draw_grid(row, "TEST INPUT", item["input"])
        draw_grid(row, "TEST PREDICTED", item["predicted"])

    root.mainloop()


if __name__ == "__main__":
    main()