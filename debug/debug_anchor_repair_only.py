import json
import os
import tkinter as tk
#  20270e3b
from reasoning.anchor_repair_rule import (
    discover_anchor_repair_rule_for_task,
    apply_anchor_repair_rule,
    solve_pair_anchor_repair_rule,
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


def add_section(parent, title):
    tk.Label(
        parent,
        text=title,
        font=("Arial", 15, "bold"),
        anchor="w",
    ).pack(fill=tk.X, padx=10, pady=(18, 2))


def main():
    task_id = input("Task id: ").strip()
    task = load_task(task_id)

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    task_rule = discover_anchor_repair_rule_for_task(train_pairs)

    print()
    print("=" * 80)
    print(f"ANCHOR REPAIR POPUP DEBUG: {task_id}")
    print("=" * 80)
    print("TASK RULE:")
    print(task_rule)

    root = tk.Tk()
    root.title(f"ANCHOR REPAIR ONLY - {task_id}")

    tk.Label(
        root,
        text=f"ANCHOR REPAIR ONLY DEBUG: {task_id}",
        font=("Arial", 16, "bold"),
    ).pack(pady=8)

    rule_text = "TASK RULE: None"
    if task_rule is not None:
        rule_text = (
            f"TASK RULE: {task_rule.get('family')} | "
            f"type={task_rule.get('rule_type')} | "
            f"anchor={task_rule.get('anchor_color')} | "
            f"exact={task_rule.get('exact_count')}/{task_rule.get('pair_count')}"
        )

    tk.Label(
        root,
        text=rule_text,
        font=("Arial", 11),
        anchor="w",
    ).pack(fill=tk.X, padx=10)

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

    add_section(scroll_frame, "PAIR-LEVEL ANCHOR REPAIR")

    for i, pair in enumerate(train_pairs):
        result = solve_pair_anchor_repair_rule(
            pair["input"],
            pair["output"],
        )

        predicted = None
        exact = False

        if result is not None:
            predicted = result.get("predicted") or result.get("prediction")
            exact = predicted == pair["output"]

        tk.Label(
            scroll_frame,
            text=f"TRAIN PAIR {i + 1} | pair-level exact={exact}",
            font=("Arial", 13, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(12, 0))

        row = tk.Frame(scroll_frame)
        row.pack(anchor="w")

        draw_grid(row, "INPUT", pair["input"])
        draw_grid(row, "EXPECTED", pair["output"])
        draw_grid(row, "PREDICTED", predicted)

    add_section(scroll_frame, "TASK-LEVEL ANCHOR REPAIR")

    for i, pair in enumerate(train_pairs):
        predicted = apply_anchor_repair_rule(
            task_rule,
            pair["input"],
        )

        exact = predicted == pair["output"]

        tk.Label(
            scroll_frame,
            text=f"TASK-LEVEL TRAIN PAIR {i + 1} | exact={exact}",
            font=("Arial", 13, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(12, 0))

        row = tk.Frame(scroll_frame)
        row.pack(anchor="w")

        draw_grid(row, "INPUT", pair["input"])
        draw_grid(row, "EXPECTED", pair["output"])
        draw_grid(row, "PREDICTED", predicted)

    add_section(scroll_frame, "TASK-LEVEL TEST PREDICTION")

    for i, pair in enumerate(test_pairs):
        predicted = apply_anchor_repair_rule(
            task_rule,
            pair["input"],
        )

        tk.Label(
            scroll_frame,
            text=f"TEST PAIR {i}",
            font=("Arial", 13, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(12, 0))

        row = tk.Frame(scroll_frame)
        row.pack(anchor="w")

        draw_grid(row, "TEST INPUT", pair["input"])
        draw_grid(row, "TEST PREDICTED", predicted)

    root.mainloop()


if __name__ == "__main__":
    main()