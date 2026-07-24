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

    # direct path
    if os.path.exists(task_name):
        with open(task_name, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict) and "train" in raw:
            return os.path.basename(task_name), raw

        if isinstance(raw, dict) and len(raw) == 1:
            task_id = next(iter(raw))
            return task_id, raw[task_id]

    # task id inside data/data.json
    if os.path.exists(data_json):
        with open(data_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        if task_name in data:
            return task_name, data[task_name]

    # task id inside data_failures/extracted_tasks
    failure_path = os.path.join(failures_dir, task_name + ".json")

    if os.path.exists(failure_path):
        with open(failure_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict) and "train" in raw:
            return task_name, raw

        if isinstance(raw, dict) and task_name in raw:
            return task_name, raw[task_name]

        if isinstance(raw, dict) and len(raw) == 1:
            task_id = next(iter(raw))
            return task_id, raw[task_id]

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


def make_rule_text(strategy_name, task_rule):
    if task_rule is None:
        return "Chosen strategy: None"

    return "\n".join(
        [
            f"Chosen strategy: {strategy_name}",
            f"Rule family: {task_rule.get('family')}",
            f"Rule type: {task_rule.get('rule_type')}",
            f"Inner strategy: {task_rule.get('chosen_inner_strategy')}",
        ]
    )


def predict_hidden_pair(train_pairs, hidden_index):
    visible_pairs = [
        pair
        for i, pair in enumerate(train_pairs)
        if i != hidden_index
    ]

    hidden_pair = train_pairs[hidden_index]

    choice = choose_task_level_strategy(visible_pairs, debug=False)

    strategy_name = choice.get("best_strategy")
    task_rule = choice.get("task_rule") or choice.get("rule")

    predicted = apply_task_rule_to_input(
        strategy_name=strategy_name,
        task_rule=task_rule,
        input_grid=hidden_pair["input"],
        expected_grid=hidden_pair["output"],
        pair_index=hidden_index,
    )

    return {
        "strategy_name": strategy_name,
        "task_rule": task_rule,
        "input": hidden_pair["input"],
        "expected": hidden_pair["output"],
        "predicted": predicted,
        "exact": predicted == hidden_pair["output"],
    }


def predict_test_pair(train_pairs, test_pair, test_index):
    choice = choose_task_level_strategy(train_pairs, debug=False)

    strategy_name = choice.get("best_strategy")
    task_rule = choice.get("task_rule") or choice.get("rule")

    predicted = apply_task_rule_to_input(
        strategy_name=strategy_name,
        task_rule=task_rule,
        input_grid=test_pair["input"],
        expected_grid=None,
        pair_index=test_index,
    )

    return {
        "strategy_name": strategy_name,
        "task_rule": task_rule,
        "input": test_pair["input"],
        "predicted": predicted,
    }


def main():
    task_name = input("Enter task id or json path: ").strip()
    task_id, task = load_task(task_name)

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    hidden_results = []

    for hidden_index in range(len(train_pairs)):
        print(f"Running hidden train pair {hidden_index + 1}...")
        hidden_results.append(
            predict_hidden_pair(train_pairs, hidden_index)
        )

    test_results = []

    for test_index, test_pair in enumerate(test_pairs):
        print(f"Running test pair {test_index}...")
        test_results.append(
            predict_test_pair(train_pairs, test_pair, test_index)
        )

    honest_exact_count = sum(1 for item in hidden_results if item["exact"])
    honest_total = len(hidden_results)

    root = tk.Tk()
    root.title(f"Honest Leave-One-Out Review - {task_id}")

    summary = tk.Label(
        root,
        text=(
            f"TASK: {task_id}\n"
            f"HONEST HIDDEN TRAIN EXACT: {honest_exact_count}/{honest_total}\n"
            f"Each train prediction was made after hiding that train pair."
        ),
        font=("Consolas", 13, "bold"),
        justify=tk.LEFT,
        anchor="w",
    )
    summary.pack(fill=tk.X, padx=10, pady=10)

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

    for i, result in enumerate(hidden_results):
        row_label = tk.Label(
            scroll_frame,
            text=(
                f"HIDDEN TRAIN PAIR {i + 1}    "
                f"exact={result['exact']}    "
                f"strategy={result['strategy_name']}    "
                f"inner={result['task_rule'].get('chosen_inner_strategy') if result['task_rule'] else None}"
            ),
            font=("Arial", 13, "bold"),
            anchor="w",
        )
        row_label.pack(fill=tk.X, padx=10, pady=(14, 0))

        rule_label = tk.Label(
            scroll_frame,
            text=make_rule_text(result["strategy_name"], result["task_rule"]),
            font=("Consolas", 10),
            justify=tk.LEFT,
            anchor="w",
        )
        rule_label.pack(fill=tk.X, padx=20, pady=(0, 4))

        row = tk.Frame(scroll_frame)
        row.pack(anchor="w")

        draw_grid(row, result["input"], "HIDDEN INPUT")
        draw_grid(row, result["expected"], "EXPECTED")
        draw_grid(row, result["predicted"], "PREDICTED")

    for i, result in enumerate(test_results):
        row_label = tk.Label(
            scroll_frame,
            text=(
                f"TEST PAIR {i}    "
                f"strategy={result['strategy_name']}    "
                f"inner={result['task_rule'].get('chosen_inner_strategy') if result['task_rule'] else None}"
            ),
            font=("Arial", 13, "bold"),
            anchor="w",
        )
        row_label.pack(fill=tk.X, padx=10, pady=(20, 0))

        rule_label = tk.Label(
            scroll_frame,
            text=make_rule_text(result["strategy_name"], result["task_rule"]),
            font=("Consolas", 10),
            justify=tk.LEFT,
            anchor="w",
        )
        rule_label.pack(fill=tk.X, padx=20, pady=(0, 4))

        row = tk.Frame(scroll_frame)
        row.pack(anchor="w")

        draw_grid(row, result["input"], "TEST INPUT")
        draw_grid(row, result["predicted"], "TEST PREDICTION")

    root.mainloop()


if __name__ == "__main__":
    main()