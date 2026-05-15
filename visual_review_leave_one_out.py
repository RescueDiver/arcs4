# visual_review_leave_one_out.py
"""
HONEST Visual Review Tool — Leave-One-Out

Purpose:
    Show what the program actually predicts when it does NOT get to
    learn from the train pair it is being tested on.

For each task and each train pair:

    1. Hide that train pair.
    2. Learn from the other train pairs.
    3. Predict the hidden input.
    4. Show:

        INPUT | EXPECTED | GUESSED

This is much more honest than normal train replay.

How to run:
    python visual_review_leave_one_out.py

Then type:
    data

or:
    data_failures/extracted_tasks/2d0172a1.json
"""

import contextlib
import io
import json
import os
import tkinter as tk
from tkinter import ttk
from reasoning.archive.ring_blob_scene import learn_ring_blob_scene
from reasoning.archive.ring_blob_rule_synthesizer import scene_signature

# from reasoning.task_router import (
#     choose_task_level_strategy,
#     apply_task_rule_to_input,
# )

from reasoning.archive.ring_blob_rule_synthesizer import (
    learn_ring_blob_rule_synthesizer,
    predict_with_ring_blob_rule_synthesizer,
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


TASK_LEVEL_STRATEGIES = {
    "multi_seed_composition_rule",
    "learned_region_rule",
    "ring_blob_rule_synthesizer",
}


# ============================================================
# FILE LOADING
# ============================================================

def load_json_file(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_task_file(user_text):
    base_dir = os.path.dirname(__file__)
    user_text = user_text.strip()

    if not user_text:
        user_text = "data"

    if user_text == "data":
        return os.path.join(base_dir, "data", "data.json")

    if user_text.endswith(".json"):
        if os.path.isabs(user_text):
            return user_text

        candidate = os.path.join(base_dir, user_text)

        if os.path.exists(candidate):
            return candidate

        candidate = os.path.join(base_dir, "data", user_text)

        if os.path.exists(candidate):
            return candidate

        return os.path.join(base_dir, user_text)

    candidate = os.path.join(base_dir, "data", user_text + ".json")

    if os.path.exists(candidate):
        return candidate

    return os.path.join(base_dir, user_text)


def normalize_loaded_tasks(raw_data):
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


def make_leave_one_out_pairs(train_pairs, held_out_index):
    learn_pairs = []

    for index, pair in enumerate(train_pairs):
        if index == held_out_index:
            continue

        learn_pairs.append(pair)

    hidden_pair = train_pairs[held_out_index]

    return learn_pairs, hidden_pair


# ============================================================
# HONEST PREDICTION
# ============================================================

def predict_leave_one_out(train_pairs, held_out_index):
    """
    Honest ring/blob-only leave-one-out prediction.

    This bypasses the task router on purpose.

    Test:
        - hide one train pair
        - learn ring_blob_rule_synthesizer from the other pairs
        - predict hidden input
        - compare to hidden expected output
    """
    learn_pairs, hidden_pair = make_leave_one_out_pairs(
        train_pairs,
        held_out_index,
    )

    hidden_input = hidden_pair["input"]
    hidden_expected = hidden_pair["output"]

    # --------------------------------------------------------
    # Print what the program sees in the hidden input.
    # --------------------------------------------------------
    hidden_scene = learn_ring_blob_scene(hidden_input)
    hidden_signature = scene_signature(hidden_scene)

    print()
    print("RAW HIDDEN SCENE KEYS")
    print(hidden_scene.keys())

    print()
    print("RAW HIDDEN SCENE")
    for key, value in hidden_scene.items():
        if key in [
            "ring_count",
            "blob_count",
            "outside_blob_count",
            "ring_blob_counts",
            "ring_layouts",
            "rings",
            "blobs",
            "blobs_by_ring",
            "outside_blobs",
            "ring_summaries",
            "ring_items",
        ]:
            print(f"{key}: {value}")

    print()
    print(f"HIDDEN PAIR {held_out_index + 1} SIGNATURE")
    print(f"  ring_count         : {hidden_signature.get('ring_count')}")
    print(f"  blob_count         : {hidden_signature.get('blob_count')}")
    print(f"  outside_blob_count : {hidden_signature.get('outside_blob_count')}")
    print(f"  ring_blob_counts   : {hidden_signature.get('ring_blob_counts')}")
    print(f"  ring_layouts       : {hidden_signature.get('ring_layouts')}")

    quiet_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(quiet_buffer):
            learned_rule = learn_ring_blob_rule_synthesizer(
                learn_pairs,
            )

            prediction_result = predict_with_ring_blob_rule_synthesizer(
                learned_rule,
                hidden_input,
            )

        predicted = prediction_result.get("prediction")

        learned_shape_rule = prediction_result.get("learned_shape_rule")
        symbolic_shape = prediction_result.get("symbolic_shape")

        if learned_shape_rule is None:
            print("  learned shape rule : None")
        else:
            height_formula = learned_shape_rule.get("height_formula", {})
            width_formula = learned_shape_rule.get("width_formula", {})

            print("  learned shape rule :")
            print(f"    height = {height_formula.get('description')}")
            print(f"    width  = {width_formula.get('description')}")
            print(f"  symbolic_shape     : {symbolic_shape}")

    except Exception as exc:
        predicted = None

        return {
            "strategy": f"ring_blob_error: {type(exc).__name__}",
            "predicted": predicted,
            "exact": False,
            "wrong": None,
            "visible_pair_count": len(learn_pairs),
        }

    exact = predicted == hidden_expected
    wrong = count_wrong_cells(predicted, hidden_expected)

    return {
        "strategy": "ring_blob_rule_synthesizer_ONLY",
        "predicted": predicted,
        "exact": exact,
        "wrong": wrong,
        "visible_pair_count": len(learn_pairs),
    }

# ============================================================
# DRAWING HELPERS
# ============================================================

def draw_grid(canvas, grid, x0, y0, cell_size, title):
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
        return 150, 70

    if not is_grid(grid):
        canvas.create_text(
            x0,
            label_y,
            text="Invalid grid",
            anchor="nw",
            font=("Arial", 11),
            fill="red",
        )
        return 150, 70

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
    visible_pair_count,
    x0,
    y0,
    cell_size,
):
    title = (
        f"HIDDEN TRAIN PAIR {pair_index + 1} | "
        f"learned_from={visible_pair_count} other pairs | "
        f"strategy={strategy} | exact={exact} | wrong={wrong}"
    )

    title_color = "green" if exact else "red"

    canvas.create_text(
        x0,
        y0,
        text=title,
        anchor="nw",
        font=("Arial", 13, "bold"),
        fill=title_color,
    )

    row_y = y0 + 30

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
    row_height = max(input_h, expected_h, guessed_h) + 60

    return row_width, row_height


# ============================================================
# WINDOW
# ============================================================

def create_scrollable_task_window(root, task_id, task, x, y):
    train_pairs = task.get("train", [])

    window = tk.Toplevel(root)
    window.title(f"HONEST LEAVE-ONE-OUT — TASK {task_id}")
    window.geometry(f"1350x850+{x}+{y}")

    outer = ttk.Frame(window)
    outer.pack(fill="both", expand=True)

    header = ttk.Label(
        outer,
        text=(
            f"TASK: {task_id}    "
            f"mode=LEAVE-ONE-OUT    "
            f"train_pairs={len(train_pairs)}"
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
    cell_size = 18

    task_right = 0
    task_wrong = 0

    for held_out_index, pair in enumerate(train_pairs):
        input_grid = pair.get("input")
        expected_grid = pair.get("output")

        prediction_info = predict_leave_one_out(
            train_pairs=train_pairs,
            held_out_index=held_out_index,
        )

        predicted_grid = prediction_info.get("predicted")
        strategy = prediction_info.get("strategy")
        exact = prediction_info.get("exact")
        wrong = prediction_info.get("wrong")
        visible_pair_count = prediction_info.get("visible_pair_count")

        pred_h, pred_w = grid_shape(predicted_grid)
        exp_h, exp_w = grid_shape(expected_grid)

        print(
            f"  hidden pair {held_out_index + 1}: "
            f"strategy={strategy} "
            f"exact={exact} "
            f"wrong={wrong} "
            f"pred_shape={pred_h}x{pred_w} "
            f"expected_shape={exp_h}x{exp_w}"
        )

        if exact:
            task_right += 1
        else:
            task_wrong += 1
        row_width, row_height = draw_pair_row(
            canvas=canvas,
            pair_index=held_out_index,
            input_grid=input_grid,
            expected_grid=expected_grid,
            predicted_grid=predicted_grid,
            strategy=strategy,
            exact=exact,
            wrong=wrong,
            visible_pair_count=visible_pair_count,
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

    print(
        f"TASK {task_id}: "
        f"leave-one-out right={task_right} "
        f"wrong={task_wrong}"
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
    print("Opening honest leave-one-out popup windows...")
    print("Each guessed output is predicted after hiding that train pair.")
    print("This should NOT be all correct unless the rule actually generalizes.")

    root = tk.Tk()
    root.withdraw()

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

        root.update_idletasks()
        root.update()

    root.mainloop()


if __name__ == "__main__":
    main()