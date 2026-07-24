# visual_review_tasks.py
"""
Visual Task Review Tool

Purpose:
    Open visual popup windows showing:

        INPUT | EXPECTED | GUESSED

    for every train pair in every task.

Important:
    This opens the windows all together.
    You do NOT need to close one window before the next appears.

How to run:
    python visual_review_tasks.py

Then type:
    data

or:
    data_failures/extracted_tasks/2d0172a1.json

or any task JSON file path.

Notes:
    - One popup window is created per task.
    - Each task window contains all train pairs for that task.
    - Each train pair shows INPUT / EXPECTED / GUESSED side by side.
"""

import json
import os
import tkinter as tk
from tkinter import ttk

from reasoning.task_router import (
    choose_task_level_strategy,
    solve_pair_with_multiple_strategies,
    apply_task_rule_to_input,
)


# ============================================================
# COLOR MAP
# ============================================================

ARC_COLORS = {
    0: "#000000",  # black
    1: "#0074D9",  # blue
    2: "#FF4136",  # red
    3: "#2ECC40",  # green
    4: "#FFDC00",  # yellow
    5: "#AAAAAA",  # gray
    6: "#F012BE",  # magenta
    7: "#FF851B",  # orange
    8: "#7FDBFF",  # light blue
    9: "#870C25",  # dark red
}


# ============================================================
# FILE LOADING
# ============================================================

def load_json_file(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_task_file(user_text):
    """
    Allow easy input like:
        data
        fc7.json
        data_failures/extracted_tasks/2d0172a1.json
    """
    base_dir = os.path.dirname(__file__)
    user_text = user_text.strip()

    if not user_text:
        user_text = "data"

    if user_text == "data":
        return os.path.join(base_dir, "data", "data.json")

    if user_text.endswith(".json"):
        # Absolute path.
        if os.path.isabs(user_text):
            return user_text

        # Relative path from project root.
        candidate = os.path.join(base_dir, user_text)

        if os.path.exists(candidate):
            return candidate

        # Relative path from data folder.
        candidate = os.path.join(base_dir, "data", user_text)

        if os.path.exists(candidate):
            return candidate

        return os.path.join(base_dir, user_text)

    # If user types a task name without .json.
    candidate = os.path.join(base_dir, "data", user_text + ".json")

    if os.path.exists(candidate):
        return candidate

    return os.path.join(base_dir, user_text)


def normalize_loaded_tasks(raw_data):
    """
    Accept either:

        {
            "task_id": {
                "train": [...],
                "test": [...]
            }
        }

    or a single task object:

        {
            "train": [...],
            "test": [...]
        }
    """
    if not isinstance(raw_data, dict):
        return {}

    if "train" in raw_data:
        return {
            "single_task": raw_data,
        }

    return raw_data


# ============================================================
# GRID HELPERS
# ============================================================

def grid_shape(grid):
    if grid is None:
        return 0, 0

    if not grid:
        return 0, 0

    return len(grid), len(grid[0])


def is_grid(grid):
    if not isinstance(grid, list):
        return False

    if not grid:
        return False

    if not all(isinstance(row, list) for row in grid):
        return False

    width = len(grid[0])

    if width == 0:
        return False

    for row in grid:
        if len(row) != width:
            return False

    return True


def count_wrong_cells(predicted, expected):
    if predicted is None or expected is None:
        return None

    ph, pw = grid_shape(predicted)
    eh, ew = grid_shape(expected)

    max_h = max(ph, eh)
    max_w = max(pw, ew)

    wrong = 0

    for r in range(max_h):
        for c in range(max_w):
            if r >= ph or c >= pw:
                wrong += 1
                continue

            if r >= eh or c >= ew:
                wrong += 1
                continue

            if predicted[r][c] != expected[r][c]:
                wrong += 1

    return wrong


# ============================================================
# PREDICTION HELPERS
# ============================================================

TASK_LEVEL_STRATEGIES = {
    "multi_seed_composition_rule",
    "learned_region_rule",
    "ring_blob_rule_synthesizer",
}


def predict_train_pair(task_router_result, input_grid, expected_grid, pair_index):
    """
    Predict one train pair using the same kind of routing main.py uses.

    If a task-level rule was chosen:
        apply the learned task rule.

    Otherwise:
        fall back to pair-level strategies.
    """
    strategy_name = task_router_result.get("best_strategy")
    task_rule = task_router_result.get("task_rule")

    if strategy_name in TASK_LEVEL_STRATEGIES and task_rule is not None:
        predicted = apply_task_rule_to_input(
            strategy_name=strategy_name,
            task_rule=task_rule,
            input_grid=input_grid,
            expected_grid=expected_grid,
            pair_index=pair_index,
        )

        exact = predicted == expected_grid
        wrong = count_wrong_cells(predicted, expected_grid)

        return {
            "strategy": strategy_name,
            "predicted": predicted,
            "exact": exact,
            "wrong": wrong,
        }

    result = solve_pair_with_multiple_strategies(
        input_grid,
        expected_grid,
        debug=False,
    )

    if result is None:
        return {
            "strategy": "no_result",
            "predicted": None,
            "exact": False,
            "wrong": None,
        }

    predicted = result.get("predicted")

    if predicted is None:
        predicted = result.get("prediction")

    exact = predicted == expected_grid
    wrong = count_wrong_cells(predicted, expected_grid)

    return {
        "strategy": result.get("strategy", "unknown"),
        "predicted": predicted,
        "exact": exact,
        "wrong": wrong,
    }


# ============================================================
# DRAWING HELPERS
# ============================================================

def draw_grid(canvas, grid, x0, y0, cell_size, title):
    """
    Draw one ARC grid onto a canvas.
    """
    canvas.create_text(
        x0,
        y0,
        text=title,
        anchor="nw",
        font=("Arial", 12, "bold"),
        fill="black",
    )

    label_y = y0 + 22

    if grid is None:
        canvas.create_text(
            x0,
            label_y,
            text="None",
            anchor="nw",
            font=("Arial", 11),
            fill="red",
        )
        return 100, 50

    if not is_grid(grid):
        canvas.create_text(
            x0,
            label_y,
            text="Invalid grid",
            anchor="nw",
            font=("Arial", 11),
            fill="red",
        )
        return 140, 50

    h, w = grid_shape(grid)

    canvas.create_text(
        x0,
        label_y,
        text=f"{h} x {w}",
        anchor="nw",
        font=("Arial", 9),
        fill="black",
    )

    grid_y = y0 + 42

    for r in range(h):
        for c in range(w):
            value = grid[r][c]
            color = ARC_COLORS.get(value, "#FFFFFF")

            x1 = x0 + c * cell_size
            y1 = grid_y + r * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=color,
                outline="#333333",
            )

    width = w * cell_size
    height = h * cell_size + 42

    return width, height


def draw_pair_row(
    canvas,
    pair_index,
    input_grid,
    expected_grid,
    predicted_grid,
    strategy,
    exact,
    wrong,
    x0,
    y0,
    cell_size,
):
    """
    Draw one train pair row:

        INPUT | EXPECTED | GUESSED
    """
    title = f"TRAIN PAIR {pair_index + 1} | strategy={strategy} | exact={exact} | wrong={wrong}"

    canvas.create_text(
        x0,
        y0,
        text=title,
        anchor="nw",
        font=("Arial", 13, "bold"),
        fill="black",
    )

    row_y = y0 + 28

    input_w, input_h = draw_grid(
        canvas,
        input_grid,
        x0,
        row_y,
        cell_size,
        "INPUT",
    )

    expected_x = x0 + input_w + 50

    expected_w, expected_h = draw_grid(
        canvas,
        expected_grid,
        expected_x,
        row_y,
        cell_size,
        "EXPECTED",
    )

    guessed_x = expected_x + expected_w + 50

    guessed_w, guessed_h = draw_grid(
        canvas,
        predicted_grid,
        guessed_x,
        row_y,
        cell_size,
        "GUESSED",
    )

    row_width = guessed_x + guessed_w - x0
    row_height = max(input_h, expected_h, guessed_h) + 55

    return row_width, row_height


# ============================================================
# SCROLLABLE TASK WINDOW
# ============================================================

def create_scrollable_task_window(root, task_id, task, x, y):
    """
    Create one popup window for one task.

    The window contains all train pairs for that task.
    """
    train_pairs = task.get("train", [])

    task_router_result = choose_task_level_strategy(
        train_pairs,
        debug=False,
    )

    chosen_strategy = task_router_result.get("best_strategy")
    has_rule = task_router_result.get("task_rule") is not None

    window = tk.Toplevel(root)
    window.title(f"TASK {task_id} — {chosen_strategy}")

    # Stagger windows slightly so they are not perfectly stacked.
    window.geometry(f"1300x850+{x}+{y}")

    outer = ttk.Frame(window)
    outer.pack(fill="both", expand=True)

    header = ttk.Label(
        outer,
        text=(
            f"TASK: {task_id}    "
            f"chosen_strategy={chosen_strategy}    "
            f"has_task_rule={has_rule}"
        ),
        font=("Arial", 13, "bold"),
    )
    header.pack(anchor="w", padx=8, pady=6)

    canvas = tk.Canvas(
        outer,
        bg="white",
        highlightthickness=0,
    )

    v_scroll = ttk.Scrollbar(
        outer,
        orient="vertical",
        command=canvas.yview,
    )

    h_scroll = ttk.Scrollbar(
        outer,
        orient="horizontal",
        command=canvas.xview,
    )

    canvas.configure(
        yscrollcommand=v_scroll.set,
        xscrollcommand=h_scroll.set,
    )

    canvas.pack(side="left", fill="both", expand=True)
    v_scroll.pack(side="right", fill="y")
    h_scroll.pack(side="bottom", fill="x")

    current_y = 10
    max_width = 1000

    # Choose a cell size that keeps big grids visible.
    cell_size = 18

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair.get("input")
        expected_grid = pair.get("output")

        prediction_info = predict_train_pair(
            task_router_result=task_router_result,
            input_grid=input_grid,
            expected_grid=expected_grid,
            pair_index=pair_index,
        )

        predicted_grid = prediction_info.get("predicted")
        strategy = prediction_info.get("strategy")
        exact = prediction_info.get("exact")
        wrong = prediction_info.get("wrong")

        row_width, row_height = draw_pair_row(
            canvas=canvas,
            pair_index=pair_index,
            input_grid=input_grid,
            expected_grid=expected_grid,
            predicted_grid=predicted_grid,
            strategy=strategy,
            exact=exact,
            wrong=wrong,
            x0=10,
            y0=current_y,
            cell_size=cell_size,
        )

        max_width = max(max_width, row_width + 30)
        current_y += row_height + 35

        canvas.create_line(
            10,
            current_y - 18,
            max_width,
            current_y - 18,
            fill="#999999",
        )

    canvas.configure(
        scrollregion=(0, 0, max_width + 100, current_y + 100),
    )

    return window


# ============================================================
# MAIN
# ============================================================

def main():
    user_text = input("Enter task file, task json, or data: ").strip()
    task_file = resolve_task_file(user_text)

    if not os.path.exists(task_file):
        print(f"File not found: {task_file}")
        return

    raw_data = load_json_file(task_file)
    tasks = normalize_loaded_tasks(raw_data)

    if not tasks:
        print("No tasks found.")
        return

    print(f"Loaded: {task_file}")
    print(f"Task count: {len(tasks)}")
    print()
    print("Opening popup windows all together...")
    print("You do not need to close one before the next appears.")

    root = tk.Tk()
    root.withdraw()

    # Stagger task windows.
    start_x = 30
    start_y = 30
    step_x = 35
    step_y = 35

    for index, (task_id, task) in enumerate(tasks.items()):
        x = start_x + (index % 8) * step_x
        y = start_y + (index % 8) * step_y

        create_scrollable_task_window(
            root=root,
            task_id=task_id,
            task=task,
            x=x,
            y=y,
        )

        # Keep UI responsive while creating many windows.
        root.update_idletasks()
        root.update()

    root.mainloop()


if __name__ == "__main__":
    main()