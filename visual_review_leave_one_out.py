# visual_review_leave_one_out.py

import json
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

from core.scoring import score_prediction

from reasoning.task_router import (
    choose_task_level_strategy,
    apply_task_rule_to_input,
)


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SHOW_POPUPS = True
PRINT_GRIDS = False
PRINT_DIFFS = True

# Important:    b0039139.json
# This file is for HONEST testing.
# The hidden output must NEVER be used while learning or predicting.


# ============================================================
# BASIC GRID HELPERS
# ============================================================

def grid_shape(grid):
    if grid is None:
        return 0, 0

    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def grids_equal(a, b):
    return a == b


def print_grid(title, grid):
    if grid is None:
        print(f"{title}: None")
        return

    h, w = grid_shape(grid)
    print(f"{title} (h={h}, w={w})")

    for row in grid:
        print(" ".join(str(v) for v in row))


def print_side_by_side(title_left, grid_left, title_right, grid_right):
    if grid_left is None or grid_right is None:
        print_grid(title_left, grid_left)
        print()
        print_grid(title_right, grid_right)
        return

    lh, lw = grid_shape(grid_left)
    rh, rw = grid_shape(grid_right)

    left_header = f"{title_left} (h={lh}, w={lw})"
    right_header = f"{title_right} (h={rh}, w={rw})"

    print(left_header.ljust(50) + right_header)

    max_h = max(lh, rh)

    for r in range(max_h):
        if r < lh:
            left_line = " ".join(str(v) for v in grid_left[r])
        else:
            left_line = ""

        if r < rh:
            right_line = " ".join(str(v) for v in grid_right[r])
        else:
            right_line = ""

        print(left_line.ljust(50) + right_line)


def count_wrong_cells(predicted, expected):
    if predicted is None or expected is None:
        return None

    ph, pw = grid_shape(predicted)
    eh, ew = grid_shape(expected)

    wrong = 0

    common_h = min(ph, eh)
    common_w = min(pw, ew)

    for r in range(common_h):
        for c in range(common_w):
            if predicted[r][c] != expected[r][c]:
                wrong += 1

    # Count extra/missing cells as wrong.
    predicted_area = ph * pw
    expected_area = eh * ew
    common_area = common_h * common_w

    wrong += (predicted_area - common_area)
    wrong += (expected_area - common_area)

    return wrong


def print_diff_summary(predicted, expected, max_show=50):
    print("\nDIFF SUMMARY")
    print("-" * 60)

    if predicted is None:
        print("Predicted is None.")
        return

    if expected is None:
        print("Expected is None.")
        return

    ph, pw = grid_shape(predicted)
    eh, ew = grid_shape(expected)

    print(f"Predicted shape: {ph}x{pw}")
    print(f"Expected shape : {eh}x{ew}")

    if ph != eh or pw != ew:
        print("Shape mismatch : yes")
    else:
        print("Shape mismatch : no")

    diffs = []

    for r in range(min(ph, eh)):
        for c in range(min(pw, ew)):
            pv = predicted[r][c]
            ev = expected[r][c]

            if pv != ev:
                diffs.append((r, c, pv, ev))

    wrong = count_wrong_cells(predicted, expected)

    print(f"Wrong cells    : {wrong}")
    print(f"Shown diffs    : {min(len(diffs), max_show)}")

    for r, c, pv, ev in diffs[:max_show]:
        print(f"  r={r:02d} c={c:02d} predicted={pv} expected={ev}")

    if len(diffs) > max_show:
        print(f"  ... {len(diffs) - max_show} more")


# ============================================================
# POPUP VIEWER
# ============================================================

def show_grids_popup(input_grid, expected_grid, predicted_grid, title_prefix):
    if not SHOW_POPUPS:
        return

    arc_colors = [
        "#000000",  # 0 black
        "#0074D9",  # 1 blue
        "#FF4136",  # 2 red
        "#2ECC40",  # 3 green
        "#FFDC00",  # 4 yellow
        "#AAAAAA",  # 5 gray
        "#F012BE",  # 6 magenta
        "#FF851B",  # 7 orange
        "#7FDBFF",  # 8 light blue
        "#870C25",  # 9 dark red / brown
    ]

    cmap = ListedColormap(arc_colors)
    norm = BoundaryNorm(np.arange(-0.5, 10.5, 1), cmap.N)

    grids = [
        ("Hidden Input", input_grid),
        ("Hidden Expected", expected_grid),
        ("Predicted", predicted_grid),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title_prefix, fontsize=16)

    for ax, (name, grid) in zip(axes, grids):
        if grid is None:
            ax.set_title(f"{name}\nNone")
            ax.axis("off")
            continue

        arr = np.array(grid)
        ax.imshow(arr, cmap=cmap, norm=norm)

        h, w = arr.shape

        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)

    plt.tight_layout()
    plt.show()


# ============================================================
# TASK LOADING
# ============================================================

def normalize_task_id_or_path(user_text):
    text = user_text.strip().strip('"').strip("'")

    if text.lower() == "data":
        return text

    return text


def candidate_task_paths(user_text):
    """
    Lets you type:
        2d0172a1
        2d0172a1.json
        data_failures/extracted_tasks/2d0172a1.json
        data/2d0172a1.json
    """
    text = normalize_task_id_or_path(user_text)

    candidates = []

    # Direct path as typed.
    candidates.append(os.path.join(BASE_DIR, text))

    # Direct path + .json.
    if not text.endswith(".json"):
        candidates.append(os.path.join(BASE_DIR, text + ".json"))

    # Task id search.
    task_id = text
    task_id = task_id.replace("\\", "/").split("/")[-1]
    if task_id.endswith(".json"):
        task_id = task_id[:-5]

    exact_filename = f"{task_id}.json"

    candidates.extend([
        os.path.join(BASE_DIR, "data_failures", "extracted_tasks", exact_filename),
        os.path.join(BASE_DIR, "data_failures", exact_filename),
        os.path.join(BASE_DIR, "data", exact_filename),
    ])

    # Remove duplicates while preserving order.
    clean = []
    seen = set()

    for path in candidates:
        norm = os.path.normpath(path)

        if norm in seen:
            continue

        seen.add(norm)
        clean.append(norm)

    return clean


def unwrap_task(raw, fallback_id=None):
    """
    Supports:
        {"train": [...], "test": [...]}

    and:
        {"task_id": {"train": [...], "test": [...]}}

    and:
        [{"train": [...], "test": [...]}]
    """
    if isinstance(raw, dict) and ("train" in raw or "test" in raw):
        return [("unknown", raw)]

    if isinstance(raw, dict):
        tasks = []

        for task_id, value in raw.items():
            if isinstance(value, dict) and ("train" in value or "test" in value):
                tasks.append((task_id, value))

        if tasks:
            return tasks

    if isinstance(raw, list):
        tasks = []

        for idx, value in enumerate(raw):
            if isinstance(value, dict) and ("train" in value or "test" in value):
                task_id = value.get("id", f"task_{idx}")
                tasks.append((task_id, value))

        if tasks:
            return tasks

    raise ValueError("Could not recognize ARC task format.")


def load_tasks_from_user_text(user_text):
    paths = candidate_task_paths(user_text)

    found_path = None

    for path in paths:
        if os.path.exists(path):
            found_path = path
            break

    if found_path is None:
        print("File not found. Tried:")
        for path in paths:
            print(f"  {path}")
        return None, []

    with open(found_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    tasks = unwrap_task(raw)

    return found_path, tasks


# ============================================================
# HONEST LEAVE-ONE-OUT
# ============================================================

def build_training_subset(train_pairs, hidden_index):
    """
    Return all train pairs EXCEPT the hidden one.

    This is the most important part of the file.

    The hidden pair's expected output must not be used while learning.
    """
    subset = []

    for idx, pair in enumerate(train_pairs):
        if idx == hidden_index:
            continue

        subset.append(pair)

    return subset


def learn_from_visible_pairs_only(visible_train_pairs):
    """
    Learn task rule from visible train pairs only.

    This calls the normal task router, but the hidden pair is absent.
    """
    task_choice = choose_task_level_strategy(
        visible_train_pairs,
        debug=True,
    )

    return task_choice


def predict_hidden_input(task_choice, hidden_input):
    """
    Predict hidden output.

    expected_grid is intentionally None.

    If a strategy needs expected_grid to work, it is not a valid test-time rule.
    """
    chosen_strategy = task_choice.get("best_strategy")
    task_rule = task_choice.get("task_rule") or task_choice.get("rule")

    if chosen_strategy is None:
        return None, chosen_strategy, task_rule

    predicted = apply_task_rule_to_input(
        strategy_name=chosen_strategy,
        task_rule=task_rule,
        input_grid=hidden_input,
        expected_grid=None,   # DO NOT LEAK HIDDEN ANSWER
        pair_index=None,
    )

    return predicted, chosen_strategy, task_rule


def print_task_choice(task_choice):
    chosen_strategy = task_choice.get("best_strategy")
    task_rule = task_choice.get("task_rule") or task_choice.get("rule")
    strategy_stats = task_choice.get("strategy_stats", {})

    print("\nLEARNED FROM VISIBLE PAIRS ONLY")
    print("-" * 60)
    print(f"Chosen strategy: {chosen_strategy}")

    if task_rule is None:
        print("Task rule      : None")
    else:
        print("Task rule      : yes")
        print(f"Family         : {task_rule.get('family')}")
        print(f"Rule type      : {task_rule.get('rule_type')}")

        if "exact_count" in task_rule and "pair_count" in task_rule:
            print(
                f"Visible exact  : "
                f"{task_rule.get('exact_count')} / {task_rule.get('pair_count')}"
            )

    if strategy_stats:
        print("\nVisible strategy stats")
        print("-" * 60)

        rows = sorted(
            strategy_stats.items(),
            key=lambda item: (
                item[1].get("exact_count", 0),
                item[1].get("total_adjusted_score", 0),
            ),
            reverse=True,
        )

        for name, stats in rows:
            print(
                f"{name:<32} "
                f"exact={stats.get('exact_count', 0):<3} "
                f"pairs={stats.get('pair_count', 0):<3} "
                f"total_adj={stats.get('total_adjusted_score', 0)}"
            )


def run_leave_one_out_for_task(task_id, task):
    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    print("\n" + "=" * 70)
    print(f"TASK {task_id}")
    print("=" * 70)
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Test pairs : {len(test_pairs)}")

    if len(train_pairs) < 2:
        print("Not enough train pairs for leave-one-out.")
        return {
            "task_id": task_id,
            "right": 0,
            "wrong": len(train_pairs),
            "total": len(train_pairs),
        }

    right = 0
    wrong = 0

    for hidden_index, hidden_pair in enumerate(train_pairs):
        hidden_input = hidden_pair["input"]
        hidden_expected = hidden_pair["output"]

        visible_train_pairs = build_training_subset(
            train_pairs=train_pairs,
            hidden_index=hidden_index,
        )

        print("\n" + "=" * 70)
        print(f"HIDDEN PAIR {hidden_index + 1}")
        print("=" * 70)
        print(f"Visible train pairs: {len(visible_train_pairs)}")
        print("Hidden expected was NOT used for learning.")
        print("Hidden expected will ONLY be used for scoring.")

        try:
            task_choice = learn_from_visible_pairs_only(visible_train_pairs)
            print_task_choice(task_choice)

            predicted, chosen_strategy, task_rule = predict_hidden_input(
                task_choice=task_choice,
                hidden_input=hidden_input,
            )

            exact = grids_equal(predicted, hidden_expected)
            score = score_prediction(predicted, hidden_expected)
            wrong_cells = count_wrong_cells(predicted, hidden_expected)

            ph, pw = grid_shape(predicted)
            eh, ew = grid_shape(hidden_expected)

            print("\nHONEST HIDDEN RESULT")
            print("-" * 60)
            print(f"Strategy       : {chosen_strategy}")
            print(f"Exact          : {exact}")
            print(f"Score          : {score}")
            print(f"Wrong cells    : {wrong_cells}")
            print(f"Pred shape     : {ph}x{pw}")
            print(f"Expected shape : {eh}x{ew}")

            if exact:
                right += 1
            else:
                wrong += 1

            if PRINT_DIFFS:
                print_diff_summary(predicted, hidden_expected)

            if PRINT_GRIDS:
                print("\nSIDE BY SIDE")
                print("-" * 60)
                print_side_by_side("EXPECTED", hidden_expected, "PREDICTED", predicted)

            try:
                show_grids_popup(
                    input_grid=hidden_input,
                    expected_grid=hidden_expected,
                    predicted_grid=predicted,
                    title_prefix=(
                        f"{task_id} — HIDDEN PAIR {hidden_index + 1} — "
                        f"exact={exact}"
                    ),
                )
            except Exception as popup_error:
                print("[POPUP ERROR]")
                print(popup_error)

        except Exception:
            wrong += 1

            print("\nERROR WHILE TESTING HIDDEN PAIR")
            print("-" * 60)
            traceback.print_exc()

    total = right + wrong

    print("\n" + "=" * 70)
    print(f"TASK {task_id} LEAVE-ONE-OUT SUMMARY")
    print("=" * 70)
    print(f"Right: {right}")
    print(f"Wrong: {wrong}")

    if total:
        print(f"Percent right: {(right / total) * 100:.2f}%")

    return {
        "task_id": task_id,
        "right": right,
        "wrong": wrong,
        "total": total,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    user_text = input("Enter task file, task json, or task id: ").strip()

    loaded_path, tasks = load_tasks_from_user_text(user_text)

    if not tasks:
        return

    print(f"Loaded: {loaded_path}")
    print(f"Task count: {len(tasks)}")

    print("\nOpening honest leave-one-out popup windows...")
    print("Each hidden output is removed before learning.")
    print("Hidden expected outputs are used ONLY for scoring.")
    print("This prevents regurgitating the answer.")

    all_right = 0
    all_wrong = 0
    all_total = 0

    for task_id, task in tasks:
        result = run_leave_one_out_for_task(task_id, task)

        all_right += result["right"]
        all_wrong += result["wrong"]
        all_total += result["total"]

    print("\n" + "=" * 70)
    print("FINAL HONEST LEAVE-ONE-OUT SUMMARY")
    print("=" * 70)
    print(f"Tasks checked: {len(tasks)}")
    print(f"Right        : {all_right}")
    print(f"Wrong        : {all_wrong}")

    if all_total:
        print(f"Percent right: {(all_right / all_total) * 100:.2f}%")

    print("\nRule trust rule:")
    print("  A strategy is NOT trusted just because it solves visible train pairs.")
    print("  It is trusted only when it predicts hidden train outputs correctly.")


if __name__ == "__main__":
    main()