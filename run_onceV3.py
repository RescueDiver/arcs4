# run_onceV4.py
"""
Clean one-task runner for visual_symbolic_ruleV2.py

Goal:
    Test the new small structure-first V2 rule family.

This runner:
    - loads one ARC task
    - runs full-train check
    - runs honest leave-one-out check
    - prints predicted vs expected for failures
    - does NOT run router
    - does NOT run old visual_symbolic_rule.py
"""

import json
import os
from copy import deepcopy
from reasoning.visual_symbolic_ruleV2 import explain_visual_symbolic_prediction
from core.scoring import score_prediction

from reasoning.visual_symbolic_ruleV2 import (
    discover_visual_symbolic_rule_v2_for_task,
    apply_visual_symbolic_rule_v2,
)


# ============================================================
# SETTINGS
# ============================================================

TARGET_TASK_FILE = "data_failures/extracted_tasks/2d0172a1.json"

RUN_FULL_TRAIN = True
RUN_LEAVE_ONE_OUT = True
RUN_TEST_PREDICTIONS = True

SHOW_FAILED_GRIDS = True
SHOW_POPUPS = True

# ============================================================
# BASIC HELPERS
# ============================================================

def grid_shape(grid):
    if grid is None:
        return 0, 0

    if not grid:
        return 0, 0

    return len(grid), len(grid[0])


def print_grid(title, grid):
    h, w = grid_shape(grid)
    print(f"{title} (h={h}, w={w})")

    if grid is None:
        print("<None>")
        return

    for row in grid:
        print(" ".join(str(x) for x in row))


def color_to_rgb(value):
    """
    ARC color palette.
    """
    palette = {
        0: "#000000",  # black
        1: "#0074D9",  # blue
        2: "#FF4136",  # red
        3: "#2ECC40",  # green
        4: "#FFDC00",  # yellow
        5: "#AAAAAA",  # gray
        6: "#F012BE",  # magenta
        7: "#FF851B",  # orange
        8: "#7FDBFF",  # cyan
        9: "#870C25",  # maroon
    }

    return palette.get(value, "#FFFFFF")


def popup_grids(title, panels):
    """
    Stable review popup.

    Opens one popup at a time.
    Close the window to continue to the next popup.
    """
    if not SHOW_POPUPS:
        return

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except Exception as e:
        print(f"[POPUP SKIPPED] matplotlib error: {e}")
        return

    colors = [color_to_rgb(i) for i in range(10)]
    cmap = ListedColormap(colors)

    count = len(panels)

    fig, axes = plt.subplots(1, count, figsize=(4 * count, 4))

    if count == 1:
        axes = [axes]

    fig.suptitle(title)

    for ax, (panel_title, grid) in zip(axes, panels):
        ax.set_title(panel_title)

        if grid is None:
            ax.text(0.5, 0.5, "None", ha="center", va="center")
            ax.axis("off")
            continue

        ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)

        h, w = grid_shape(grid)

        ax.set_xticks([x - 0.5 for x in range(1, w)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, h)], minor=True)
        ax.grid(which="minor", color="black", linewidth=0.5)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    print()
    print(f"[POPUP] {title}")
    print("Close the popup window to continue...")

    plt.show(block=True)
    plt.close(fig)


def normalize_train_pairs(task):
    pairs = []

    for pair in task.get("train", []):
        if "input" not in pair or "output" not in pair:
            continue

        pairs.append({
            "input": pair["input"],
            "output": pair["output"],
        })

    return pairs


def normalize_test_pairs(task):
    pairs = []

    for pair in task.get("test", []):
        if "input" not in pair:
            continue

        item = {
            "input": pair["input"],
        }

        if "output" in pair:
            item["output"] = pair["output"]

        pairs.append(item)

    return pairs


def load_task(path):
    here = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(here, path)

    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 1: file is already one ARC task
    if isinstance(data, dict) and "train" in data and "test" in data:
        return data

    # Case 2: file is wrapped by task id
    if isinstance(data, dict):
        if "2d0172a1" in data:
            return data["2d0172a1"]

        # fallback: take the first task-looking value
        for value in data.values():
            if isinstance(value, dict) and "train" in value and "test" in value:
                return value

    raise ValueError(f"Could not find ARC task in file: {path}")


def diff_cells(predicted, expected):
    diffs = []

    ph, pw = grid_shape(predicted)
    eh, ew = grid_shape(expected)

    if ph != eh or pw != ew:
        return [{
            "shape_mismatch": True,
            "pred_shape": (ph, pw),
            "expected_shape": (eh, ew),
        }]

    for r in range(eh):
        for c in range(ew):
            if predicted[r][c] != expected[r][c]:
                diffs.append({
                    "row": r,
                    "col": c,
                    "pred": predicted[r][c],
                    "expected": expected[r][c],
                })

    return diffs


def print_diff_summary(label, predicted, expected):
    ph, pw = grid_shape(predicted)
    eh, ew = grid_shape(expected)

    print()
    print(f"[{label} DIFF SUMMARY]")
    print(f"pred shape: {ph}x{pw}")
    print(f"exp  shape: {eh}x{ew}")

    diffs = diff_cells(predicted, expected)

    if diffs and diffs[0].get("shape_mismatch"):
        print("shape mismatch")
    else:
        print(f"wrong cells: {len(diffs)}")

    if SHOW_FAILED_GRIDS and diffs:
        print()
        print_grid("PREDICTED GRID", predicted)
        print()
        print_grid("EXPECTED GRID", expected)

    for diff in diffs[:80]:
        if diff.get("shape_mismatch"):
            continue

        print(
            f"  r={diff['row']:2d} "
            f"c={diff['col']:2d} "
            f"pred={diff['pred']} "
            f"exp={diff['expected']}"
        )


# ============================================================
# FULL TRAIN CHECK
# ============================================================

def run_full_train_check(train_pairs):
    print()
    print("=" * 60)
    print("VISUAL SYMBOLIC V2 FULL-TRAIN CHECK")
    print("=" * 60)

    rule = discover_visual_symbolic_rule_v2_for_task(train_pairs)

    exact_count = 0
    total_score = 0
    results = []

    for idx, pair in enumerate(train_pairs):
        from reasoning.visual_symbolic_ruleV2 import extract_scene_facts

        scene = extract_scene_facts(pair["input"])
        print()
        print(f"[SCENE FACTS pair {idx}]")
        print(f"ring_count          : {scene.get('ring_count')}")
        print(f"blob_count          : {scene.get('blob_count')}")
        print(f"outside_blob_count  : {scene.get('outside_blob_count')}")
        print(f"nested_ring_count   : {scene.get('nested_ring_count')}")
        print(f"ring_blob_counts    : {scene.get('ring_blob_counts')}")
        print(f"outside_blobs       : {scene.get('outside_blobs')}")
        predicted = apply_visual_symbolic_rule_v2(rule, pair["input"])
        explanation = explain_visual_symbolic_prediction(pair["input"])
        predicted = explanation["prediction"]

        print()
        print("[V2 STRUCTURAL EXPLANATION]")
        print(f"output_shape      : {explanation['output_shape']}")
        print(f"outer_frame       : {explanation['outer_frame']}")
        print(f"inner_frame       : {explanation['inner_frame']}")
        print(f"one_ring_markers  : {explanation['one_ring_markers']}")
        print(f"nested_markers    : {explanation['nested_markers']}")
        print(f"blob_geometry    : {explanation['blob_geometry']}")
        expected = pair["output"]

        score = score_prediction(predicted, expected)
        exact = predicted == expected

        if exact:
            exact_count += 1

        total_score += score

        ph, pw = grid_shape(predicted)

        results.append({
            "pair_index": idx,
            "exact": exact,
            "score": score,
            "shape": f"{ph}x{pw}",
            "predicted": predicted,
            "expected": expected,
        })

    print()
    print("[visual_symbolic_ruleV2 FULL TRAIN]")
    print("-" * 60)
    print(f"Exact count: {exact_count}")
    print(f"Pair count : {len(train_pairs)}")
    print(f"Total score: {total_score}")

    for result in results:
        print(
            f"  pair {result['pair_index']}: "
            f"exact={result['exact']} "
            f"score={result['score']} "
            f"shape={result['shape']}"
        )

    if not result["exact"]:
        print_diff_summary(
            f"FULL TRAIN PAIR {result['pair_index']}",
            result["predicted"],
            result["expected"],
        )

        popup_grids(
            f"FULL TRAIN PAIR {result['pair_index']}",
            [
                ("EXPECTED", result["expected"]),
                ("PREDICTED", result["predicted"]),
            ],
        )

    return {
        "exact_count": exact_count,
        "pair_count": len(train_pairs),
        "total_score": total_score,
        "results": results,
    }


# ============================================================
# LEAVE ONE OUT CHECK
# ============================================================

def run_leave_one_out_check(train_pairs):
    print()
    print("=" * 60)
    print("VISUAL SYMBOLIC V2 LEAVE-ONE-OUT CHECK")
    print("=" * 60)

    exact_count = 0
    total_score = 0
    results = []

    for hidden_idx in range(len(train_pairs)):
        visible_pairs = [
            deepcopy(pair)
            for idx, pair in enumerate(train_pairs)
            if idx != hidden_idx
        ]

        hidden_pair = train_pairs[hidden_idx]

        print()
        print("-" * 60)
        print(f"LOO HIDDEN PAIR {hidden_idx}")
        print("-" * 60)

        rule = discover_visual_symbolic_rule_v2_for_task(visible_pairs)

        predicted = apply_visual_symbolic_rule_v2(
            rule,
            hidden_pair["input"],
        )

        expected = hidden_pair["output"]

        score = score_prediction(predicted, expected)
        exact = predicted == expected

        if exact:
            exact_count += 1

        total_score += score

        ph, pw = grid_shape(predicted)

        results.append({
            "pair_index": hidden_idx,
            "exact": exact,
            "score": score,
            "shape": f"{ph}x{pw}",
            "predicted": predicted,
            "expected": expected,
        })

        if not exact:
            print_diff_summary(
                f"LOO HIDDEN PAIR {hidden_idx}",
                predicted,
                expected,
            )

        popup_grids(
            f"LOO HIDDEN PAIR {hidden_idx} | exact={exact}",
            [
                ("INPUT", hidden_pair["input"]),
                ("EXPECTED", expected),
                ("PREDICTED", predicted),
            ],
        )

    print()
    print("[visual_symbolic_ruleV2 LEAVE ONE OUT]")
    print("-" * 60)
    print(f"Exact count: {exact_count}")
    print(f"Pair count : {len(train_pairs)}")
    print(f"Total score: {total_score}")

    for result in results:
        print(
            f"  pair {result['pair_index']}: "
            f"exact={result['exact']} "
            f"score={result['score']} "
            f"shape={result['shape']}"
        )

    return {
        "exact_count": exact_count,
        "pair_count": len(train_pairs),
        "total_score": total_score,
        "results": results,
    }


# ============================================================
# TEST PREDICTIONS
# ============================================================

def run_test_predictions(train_pairs, test_pairs):
    print()
    print("=" * 60)
    print("VISUAL SYMBOLIC V2 TEST PREDICTIONS")
    print("=" * 60)

    rule = discover_visual_symbolic_rule_v2_for_task(train_pairs)

    from reasoning.visual_symbolic_ruleV2 import (
        extract_scene_facts,
        explain_visual_symbolic_prediction,
    )

    for idx, pair in enumerate(test_pairs):
        scene = extract_scene_facts(pair["input"])

        print()
        print(f"[TEST SCENE FACTS {idx}]")
        print(f"ring_count          : {scene.get('ring_count')}")
        print(f"blob_count          : {scene.get('blob_count')}")
        print(f"outside_blob_count  : {scene.get('outside_blob_count')}")
        print(f"nested_ring_count   : {scene.get('nested_ring_count')}")
        print(f"ring_blob_counts    : {scene.get('ring_blob_counts')}")
        print(f"outside_blobs       : {scene.get('outside_blobs')}")

        explanation = explain_visual_symbolic_prediction(pair["input"])
        predicted = explanation["prediction"]

        print()
        print("[V2 TEST STRUCTURAL EXPLANATION]")
        print(f"output_shape      : {explanation['output_shape']}")
        print(f"outer_frame       : {explanation['outer_frame']}")
        print(f"inner_frame       : {explanation['inner_frame']}")
        print(f"one_ring_markers  : {explanation['one_ring_markers']}")
        print(f"nested_markers    : {explanation['nested_markers']}")
        print(f"blob_geometry     : {explanation['blob_geometry']}")
        print(f"ring_geometry     : {explanation['ring_geometry']}")
        print(f"blob_directions   : {explanation['blob_directions']}")
        print()
        print("-" * 60)
        print(f"TEST PAIR {idx}")
        print("-" * 60)
        print_grid("INPUT", pair["input"])
        print()
        print_grid("PREDICTED", predicted)

        panels = [
            ("INPUT", pair["input"]),
            ("PREDICTED", predicted),
        ]

        if "output" in pair:
            panels.insert(1, ("EXPECTED", pair["output"]))

        popup_grids(
            (
                f"TEST PAIR {idx} | "
                f"rings={scene.get('ring_count')} "
                f"blobs={scene.get('blob_count')} "
                f"outside={scene.get('outside_blob_count')} "
                f"ring_blobs={scene.get('ring_blob_counts')} "
                f"markers={explanation['nested_markers']}"
            ),
            panels,
        )

        if "output" in pair:
            print()
            print_grid("EXPECTED", pair["output"])
            print_diff_summary(
                f"TEST PAIR {idx}",
                predicted,
                pair["output"],
            )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("RUN_ONCE_V4 — VISUAL SYMBOLIC RULE V2")
    print("=" * 60)

    task = load_task(TARGET_TASK_FILE)

    train_pairs = normalize_train_pairs(task)
    test_pairs = normalize_test_pairs(task)

    print(f"Task file  : {TARGET_TASK_FILE}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Test pairs : {len(test_pairs)}")

    full_train_result = None
    loo_result = None

    if RUN_FULL_TRAIN:
        full_train_result = run_full_train_check(train_pairs)

    if RUN_LEAVE_ONE_OUT:
        loo_result = run_leave_one_out_check(train_pairs)

    if RUN_TEST_PREDICTIONS:
        run_test_predictions(train_pairs, test_pairs)

    print()
    print("=" * 60)
    print("FINAL QUICK SUMMARY")
    print("=" * 60)

    if full_train_result is not None:
        print(
            f"Full train: "
            f"{full_train_result['exact_count']} / "
            f"{full_train_result['pair_count']}"
        )
    else:
        print("Full train: skipped")

    if loo_result is not None:
        print(
            f"LOO       : "
            f"{loo_result['exact_count']} / "
            f"{loo_result['pair_count']}"
        )
    else:
        print("LOO       : skipped")

    print("Router    : skipped")




if __name__ == "__main__":
    main()