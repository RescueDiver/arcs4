import json
import os
import tkinter as tk

from reasoning.task_router import (
    choose_task_level_strategy,
    apply_task_rule_to_input,
)


CELL = 22

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


def load_task(task_name):
    base_dir = os.path.dirname(__file__)

    data_json = os.path.join(base_dir, "data", "data.json")
    failures_dir = os.path.join(base_dir, "data_failures", "extracted_tasks")

    # direct json path
    if os.path.exists(task_name):
        with open(task_name, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if "train" in raw:
            return task_name, raw

        if isinstance(raw, dict) and len(raw) == 1:
            task_id = next(iter(raw))
            return task_id, raw[task_id]

    # data/data.json by task id
    if os.path.exists(data_json):
        with open(data_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        if task_name in data:
            return task_name, data[task_name]

    # data_failures/extracted_tasks by task id
    failure_path = os.path.join(failures_dir, task_name + ".json")
    if os.path.exists(failure_path):
        with open(failure_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if "train" in raw:
            return task_name, raw

        if task_name in raw:
            return task_name, raw[task_name]

    raise FileNotFoundError(f"Could not find task: {task_name}")


def grid_shape(grid):
    if grid is None:
        return 0, 0
    return len(grid), len(grid[0]) if grid else 0


def draw_grid(parent, grid, title):
    frame = tk.Frame(parent)
    frame.pack(side=tk.LEFT, padx=10, pady=8, anchor="n")

    label = tk.Label(frame, text=title, font=("Arial", 11, "bold"))
    label.pack()

    if grid is None:
        canvas = tk.Canvas(
            frame,
            width=180,
            height=80,
            bg="#333333",
            highlightthickness=1,
            highlightbackground="white",
        )
        canvas.pack()
        canvas.create_text(
            90,
            40,
            text="NONE",
            fill="white",
            font=("Arial", 18, "bold"),
        )
        return

    h, w = grid_shape(grid)

    canvas = tk.Canvas(
        frame,
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
                outline="#444444",
            )


def make_status_text(strategy_name, task_rule):
    if task_rule is None:
        return "Chosen strategy: None"

    lines = [
        f"Chosen strategy: {strategy_name}",
        f"Rule family: {task_rule.get('family')}",
        f"Rule type: {task_rule.get('rule_type')}",
        f"Inner strategy: {task_rule.get('chosen_inner_strategy')}",
    ]

    return "\n".join(lines)


def main():
    task_name = input("Enter task id or json path: ").strip()
    task_id, task = load_task(task_name)

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    choice = choose_task_level_strategy(train_pairs, debug=False)

    strategy_name = choice.get("best_strategy")
    task_rule = choice.get("task_rule") or choice.get("rule")

    root = tk.Tk()
    root.title(f"Visual Solver Review - {task_id}")

    top_label = tk.Label(
        root,
        text=make_status_text(strategy_name, task_rule),
        font=("Consolas", 12),
        justify=tk.LEFT,
        anchor="w",
    )
    top_label.pack(fill=tk.X, padx=10, pady=10)

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
    canvas.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    x_scroll.pack(side=tk.BOTTOM, fill=tk.X)

    for i, pair in enumerate(train_pairs):
        inp = pair["input"]
        expected = pair["output"]

        predicted = apply_task_rule_to_input(
            strategy_name=strategy_name,
            task_rule=task_rule,
            input_grid=inp,
            expected_grid=expected,
            pair_index=i,
        )

        exact = predicted == expected

        row_label = tk.Label(
            scroll_frame,
            text=f"TRAIN PAIR {i + 1}    exact={exact}",
            font=("Arial", 13, "bold"),
            anchor="w",
        )
        row_label.pack(fill=tk.X, padx=10, pady=(12, 0))

        row = tk.Frame(scroll_frame)
        row.pack(anchor="w")

        draw_grid(row, inp, "INPUT")
        draw_grid(row, expected, "EXPECTED")
        draw_grid(row, predicted, "PREDICTED")

    for i, pair in enumerate(test_pairs):
        inp = pair["input"]

        predicted = apply_task_rule_to_input(
            strategy_name=strategy_name,
            task_rule=task_rule,
            input_grid=inp,
            expected_grid=None,
            pair_index=i,
        )

        row_label = tk.Label(
            scroll_frame,
            text=f"TEST PAIR {i}",
            font=("Arial", 13, "bold"),
            anchor="w",
        )
        row_label.pack(fill=tk.X, padx=10, pady=(18, 0))

        row = tk.Frame(scroll_frame)
        row.pack(anchor="w")

        draw_grid(row, inp, "TEST INPUT")
        draw_grid(row, predicted, "TEST PREDICTION")

    root.mainloop()


if __name__ == "__main__":
    main()