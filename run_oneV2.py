# run_oneV2.py

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

from core.scoring import score_prediction

from reasoning.task_router import (
    choose_task_level_strategy,
    apply_task_rule_to_input,
    get_all_strategy_results,
)


# ============================================================
# TASK TO DEBUG
# ============================================================

TARGET_TASK_ID = "269e22fb"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SHOW_POPUPS = True
SHOW_PAIR_LEVEL_CANDIDATES = True


# ============================================================
# BASIC GRID HELPERS
# ============================================================

def grid_shape(grid):
    if grid is None:
        return 0, 0

    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def print_grid(title, grid):
    if grid is None:
        print(f"{title}: None")
        return

    h, w = grid_shape(grid)
    print(f"{title} (h={h}, w={w})")

    for row in grid:
        print(" ".join(str(v) for v in row))


def print_side_by_side(title_left, grid_left, title_right, grid_right):
    """
    Print two grids next to each other.

    Useful for quick terminal comparison.
    """
    if grid_left is None or grid_right is None:
        print_grid(title_left, grid_left)
        print()
        print_grid(title_right, grid_right)
        return

    lh, lw = grid_shape(grid_left)
    rh, rw = grid_shape(grid_right)

    print(f"{title_left} (h={lh}, w={lw})".ljust(45) + f"{title_right} (h={rh}, w={rw})")

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

        print(left_line.ljust(45) + right_line)


def print_diff_summary(predicted, expected, max_show=40):
    if predicted is None or expected is None:
        print("\nDIFF SUMMARY")
        print("-" * 60)
        print("Cannot compare because predicted or expected is None.")
        return

    ph, pw = grid_shape(predicted)
    eh, ew = grid_shape(expected)

    print("\nDIFF SUMMARY")
    print("-" * 60)
    print(f"Predicted shape: {ph}x{pw}")
    print(f"Expected shape : {eh}x{ew}")

    diffs = []

    for r in range(min(ph, eh)):
        for c in range(min(pw, ew)):
            pv = predicted[r][c]
            ev = expected[r][c]

            if pv != ev:
                diffs.append((r, c, pv, ev))

    # Count extra shape cells as shape mismatch info.
    shape_mismatch = (ph != eh or pw != ew)

    print(f"Different cells: {len(diffs)}")

    if shape_mismatch:
        print("Shape mismatch: yes")

    for r, c, pv, ev in diffs[:max_show]:
        print(f"  r={r:02d} c={c:02d} predicted={pv} expected={ev}")

    if len(diffs) > max_show:
        print(f"  ... {len(diffs) - max_show} more")


# ============================================================
# POPUP VISUALIZER
# ============================================================

def show_grids_popup(input_grid, expected_grid, predicted_grid, title_prefix="PAIR"):
    """
    Simple ARC grid popup.

    This file is only a debug viewer.
    The solving logic lives in the rule files and task_router.py.
    """
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
        ("Input", input_grid),
        ("Expected", expected_grid),
        ("Predicted", predicted_grid),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title_prefix, fontsize=16)

    for ax, (name, grid) in zip(axes, grids):
        if grid is None:
            ax.set_title(f"{name}\n(None)")
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
# TASK LOADER
# ============================================================

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
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Case 1:
        # {
        #   "269e22fb": {
        #       "train": [...],
        #       "test": [...]
        #   }
        # }
        if isinstance(raw, dict) and task_id in raw:
            return raw[task_id]

        # Case 2:
        # {
        #   "train": [...],
        #   "test": [...]
        # }
        if isinstance(raw, dict) and ("train" in raw or "test" in raw):
            return raw

        # Case 3:
        # {
        #   "some_id": {
        #       "train": [...],
        #       "test": [...]
        #   }
        # }
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
# ROUTER / STRATEGY PRINTS
# ============================================================

def print_task_choice_summary(task_choice):
    chosen_strategy = task_choice.get("best_strategy")
    task_rule = task_choice.get("task_rule") or task_choice.get("rule")
    strategy_stats = task_choice.get("strategy_stats", {})

    print("\n" + "=" * 60)
    print("TASK-LEVEL ROUTER RESULT")
    print("=" * 60)
    print(f"Chosen strategy: {chosen_strategy}")

    if task_rule is None:
        print("Learned task rule: None")
    else:
        print("Learned task rule: yes")
        print(f"Rule family      : {task_rule.get('family')}")
        print(f"Rule type        : {task_rule.get('rule_type')}")

        if "exact_count" in task_rule and "pair_count" in task_rule:
            print(f"Exact train      : {task_rule.get('exact_count')} / {task_rule.get('pair_count')}")

        if "residual_rule" in task_rule:
            residual_rule = task_rule.get("residual_rule", {})
            print(f"Residual rule    : {residual_rule.get('type')}")

            connector = residual_rule.get("connector_stamp", {})
            line = residual_rule.get("line", {})

            if connector:
                print(f"Connector stamp  : {connector.get('cell_count')} cells")

            if line:
                print(f"Line component   : {line.get('cell_count')} cells")

    print("\nSTRATEGY STATS")
    print("-" * 60)

    if not strategy_stats:
        print("No strategy stats.")
        return

    rows = sorted(
        strategy_stats.items(),
        key=lambda item: (
            item[1].get("exact_count", 0),
            item[1].get("total_adjusted_score", 0),
            item[1].get("pair_count", 0),
        ),
        reverse=True,
    )

    for strategy, stats in rows:
        print(
            f"{strategy:<30} "
            f"exact={stats.get('exact_count', 0):<3} "
            f"pairs={stats.get('pair_count', 0):<3} "
            f"total_adj={stats.get('total_adjusted_score', 0)}"
        )


def print_pair_level_candidates(input_grid, expected_grid):
    """
    Optional pair-level debug.

    This shows what the normal pair router sees.
    It is not the final task-level decision.
    """
    if not SHOW_PAIR_LEVEL_CANDIDATES:
        return

    if expected_grid is None:
        return

    print("\nPAIR-LEVEL CANDIDATES")
    print("-" * 60)

    candidates = get_all_strategy_results(
        input_grid,
        expected_grid,
        debug=False,
    )

    if not candidates:
        print("No pair-level candidates.")
        return

    candidates.sort(
        key=lambda c: (
            1 if c.get("exact") else 0,
            c.get("adjusted_score", c.get("score", 0)),
        ),
        reverse=True,
    )

    for c in candidates:
        pred = c.get("predicted")
        ph, pw = grid_shape(pred)

        print(
            f"{c.get('strategy'):<30} "
            f"raw={c.get('raw_score', c.get('score')):<7} "
            f"adj={c.get('adjusted_score', c.get('score')):<7} "
            f"exact={str(c.get('exact')):<5} "
            f"shape={ph}x{pw}"
        )


# ============================================================
# TRAIN / TEST DEBUG RUNNERS
# ============================================================

def run_train_debug(train_pairs, chosen_strategy, task_rule):
    total_right = 0
    total_wrong = 0

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        expected_grid = pair["output"]

        print("\n" + "=" * 60)
        print(f"TRAIN PAIR {pair_index + 1}")
        print("=" * 60)

        ih, iw = grid_shape(input_grid)
        eh, ew = grid_shape(expected_grid)

        print(f"Input shape   : {ih}x{iw}")
        print(f"Expected shape: {eh}x{ew}")

        print_pair_level_candidates(input_grid, expected_grid)

        predicted = apply_task_rule_to_input(
            strategy_name=chosen_strategy,
            task_rule=task_rule,
            input_grid=input_grid,
            expected_grid=expected_grid,
            pair_index=pair_index,
        )

        score = score_prediction(predicted, expected_grid)
        exact = predicted == expected_grid

        print("\nCHOSEN TASK-LEVEL RESULT")
        print("-" * 60)
        print(f"Strategy: {chosen_strategy}")
        print(f"Score   : {score}")
        print(f"Exact   : {exact}")

        print_diff_summary(predicted, expected_grid)

        print("\nSIDE BY SIDE")
        print("-" * 60)
        print_side_by_side("EXPECTED", expected_grid, "PREDICTED", predicted)

        if exact:
            total_right += 1
        else:
            total_wrong += 1

        try:
            show_grids_popup(
                input_grid,
                expected_grid,
                predicted,
                title_prefix=f"{TARGET_TASK_ID} - TRAIN PAIR {pair_index + 1}",
            )
        except Exception as e:
            print("[POPUP ERROR]")
            print(e)

    print("\n" + "=" * 60)
    print("TRAIN SUMMARY")
    print("=" * 60)

    print(f"Right: {total_right}")
    print(f"Wrong: {total_wrong}")

    total = total_right + total_wrong

    if total:
        print(f"Percent right: {(total_right / total) * 100:.2f}%")


def run_test_debug(test_pairs, chosen_strategy, task_rule):
    if not test_pairs:
        return

    print("\n" + "=" * 60)
    print("TEST PAIRS")
    print("=" * 60)

    for test_index, pair in enumerate(test_pairs):
        input_grid = pair["input"]
        expected_grid = pair.get("output")

        print("\n" + "-" * 60)
        print(f"TEST PAIR {test_index + 1}")
        print("-" * 60)

        predicted = apply_task_rule_to_input(
            strategy_name=chosen_strategy,
            task_rule=task_rule,
            input_grid=input_grid,
            expected_grid=expected_grid,
            pair_index=None,
        )

        print_grid("TEST INPUT", input_grid)
        print()
        print_grid("EXPECTED", expected_grid)
        print()
        print_grid("PREDICTED", predicted)

        if expected_grid is not None:
            score = score_prediction(predicted, expected_grid)
            exact = predicted == expected_grid

            print("\nTEST SCORE")
            print("-" * 60)
            print(f"Score: {score}")
            print(f"Exact: {exact}")

            print_diff_summary(predicted, expected_grid)

        try:
            show_grids_popup(
                input_grid,
                expected_grid,
                predicted,
                title_prefix=f"{TARGET_TASK_ID} - TEST PAIR {test_index + 1}",
            )
        except Exception as e:
            print("[POPUP ERROR]")
            print(e)


# ============================================================
# MAIN
# ============================================================

def main():
    task = load_task(TARGET_TASK_ID)

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    print("=" * 60)
    print(f"RUN_ONE_V2 — TASK {TARGET_TASK_ID}")
    print("=" * 60)
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Test pairs : {len(test_pairs)}")

    if not train_pairs:
        print("No train pairs found.")
        return

    task_choice = choose_task_level_strategy(
        train_pairs,
        debug=True,
    )

    chosen_strategy = task_choice.get("best_strategy")
    task_rule = task_choice.get("task_rule") or task_choice.get("rule")

    print_task_choice_summary(task_choice)

    if chosen_strategy is None:
        print("\nNo strategy was chosen.")
        return

    run_train_debug(
        train_pairs=train_pairs,
        chosen_strategy=chosen_strategy,
        task_rule=task_rule,
    )

    run_test_debug(
        test_pairs=test_pairs,
        chosen_strategy=chosen_strategy,
        task_rule=task_rule,
    )


if __name__ == "__main__":
    main()