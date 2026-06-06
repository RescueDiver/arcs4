# run_onceV3.py
"""
Clean one-task runner for visual_symbolic_ruleV2.py

Main flow:
    1. Learn from ALL train pairs.
    2. Apply learned marker rule back to train.
    3. Apply learned marker rule to test inputs.
    4. Run LOO last as diagnostic only.

Important:
    LOO does NOT control the final learned rule.
    LOO only tells us how dependent the rule is on seeing all train pairs.
"""

import json
import os
from copy import deepcopy

from core.scoring import score_prediction

from reasoning.visual_symbolic_ruleV2 import (
    discover_visual_symbolic_rule_v2_for_task,
    apply_visual_symbolic_rule_v2,
    explain_visual_symbolic_prediction,
    extract_scene_facts,
)

from reasoning.visual_symbolic_output_learner import (
    build_marker_learning_example,
    print_marker_learning_example,
    learn_marker_placement_rule,
    print_marker_placement_rule,
    apply_learned_marker_rule,
    print_marker_decision_report,
    source_facts_from_scene,
)
from reasoning.task_router import try_anchor_compass_merge_for_test_pair

# ============================================================
# SETTINGS
# ============================================================

TARGET_TASK_FILE = "data_failures/extracted_tasks/2d0172a1.json"


# ============================================================
# DEBUG SETTINGS
# ============================================================

SHOW_POPUPS = True

PRINT_NUMBER_GRIDS = True
PRINT_BLOB_GEOMETRY = True
PRINT_RING_GEOMETRY = True
PRINT_SCENE_FACTS = True
PRINT_MARKER_LEARNING = True
PRINT_DIFFS = True


# ============================================================
# RUN CONTROLS
# ============================================================

RUN_FULL_TRAIN = True
RUN_FRAME_ONLY_BASELINE = True
RUN_TEST_PREDICTIONS = True
RUN_LEAVE_ONE_OUT = True


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
    if not PRINT_NUMBER_GRIDS:
        return

    h, w = grid_shape(grid)
    print(f"{title} (h={h}, w={w})")

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
            ax.text(
                0.5,
                0.5,
                "None",
                ha="center",
                va="center",
            )
            ax.axis("off")
            continue

        ax.imshow(
            grid,
            cmap=cmap,
            vmin=0,
            vmax=9,
            interpolation="nearest",
        )

        h, w = grid_shape(grid)

        ax.set_xticks(range(w))
        ax.set_yticks(range(h))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, color="white", linewidth=0.5)
        ax.tick_params(length=0)

    print(f"\n[POPUP] {title}")
    print("Close the popup window to continue...")
    plt.tight_layout()
    plt.show()


def load_task(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 1: file is already one ARC task.
    if isinstance(data, dict) and "train" in data and "test" in data:
        return data

    # Case 2: file is wrapped by task id.
    if isinstance(data, dict):
        if "2d0172a1" in data:
            return data["2d0172a1"]

        # Fallback: take the first task-looking value.
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
    if not PRINT_DIFFS:
        return

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

    for diff in diffs[:80]:
        if diff.get("shape_mismatch"):
            continue

        print(
            f"  r={diff['row']:2d} "
            f"c={diff['col']:2d} "
            f"pred={diff['pred']} "
            f"exp={diff['expected']}"
        )


def print_scene_facts(label, input_grid):
    if not PRINT_SCENE_FACTS:
        return

    scene = extract_scene_facts(input_grid)

    print()
    print(f"[SCENE FACTS {label}]")
    print(f"ring_count          : {scene.get('ring_count')}")
    print(f"blob_count          : {scene.get('blob_count')}")
    print(f"outside_blob_count  : {scene.get('outside_blob_count')}")
    print(f"nested_ring_count   : {scene.get('nested_ring_count')}")
    print(f"ring_blob_counts    : {scene.get('ring_blob_counts')}")
    print(f"outside_blobs       : {scene.get('outside_blobs')}")


def print_source_facts(explanation):
    source_facts = source_facts_from_scene(explanation)

    print("source_facts:")

    for fact in source_facts:
        print(
            f"  {fact.get('source_key')} "
            f"simple={fact.get('simple_source_key')} "
            f"role={fact.get('container_role')} "
            f"direction={fact.get('direction')} "
            f"ring_blob_count={fact.get('ring_blob_count')}"
        )


def print_structural_explanation(label, explanation):
    print()
    print(f"[{label}]")
    print(f"output_shape      : {explanation.get('output_shape')}")
    print(f"outer_frame       : {explanation.get('outer_frame')}")
    print(f"inner_frame       : {explanation.get('inner_frame')}")
    print(f"one_ring_markers  : {explanation.get('one_ring_markers')}")
    print(f"nested_markers    : {explanation.get('nested_markers')}")

    if PRINT_BLOB_GEOMETRY:
        print(f"blob_geometry     : {explanation.get('blob_geometry')}")

    if PRINT_RING_GEOMETRY:
        print(f"ring_geometry     : {explanation.get('ring_geometry')}")

    print(f"blob_directions   : {explanation.get('blob_directions')}")


def build_learning_examples_from_train(train_pairs):
    learning_examples = []

    for idx, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        output_grid = pair["output"]

        explanation = explain_visual_symbolic_prediction(input_grid)

        learning_example = build_marker_learning_example(
            pair_index=idx,
            input_grid=input_grid,
            output_grid=output_grid,
            explanation=explanation,
        )

        learning_examples.append(learning_example)

    return learning_examples


def learn_marker_rule_from_train(train_pairs, should_print=True):
    learning_examples = build_learning_examples_from_train(train_pairs)

    if should_print and PRINT_MARKER_LEARNING:
        for learning_example in learning_examples:
            print_marker_learning_example(learning_example)

    marker_rule = learn_marker_placement_rule(learning_examples)

    if should_print:
        print_marker_placement_rule(marker_rule)

    return marker_rule


def predict_with_learned_markers(input_grid, marker_rule):
    explanation = explain_visual_symbolic_prediction(input_grid)

    frame_prediction = explanation["prediction"]

    learned_prediction, applied_markers = apply_learned_marker_rule(
        prediction=frame_prediction,
        explanation=explanation,
        marker_rule=marker_rule,
    )

    return learned_prediction, explanation, applied_markers


# ============================================================
# MAIN FULL-TRAIN LEARNED-MARKER CHECK
# ============================================================

def run_full_train_learned_marker_check(train_pairs):
    print()
    print("=" * 60)
    print("MAIN CHECK — LEARN FROM ALL TRAIN PAIRS")
    print("=" * 60)

    marker_rule = learn_marker_rule_from_train(
        train_pairs=train_pairs,
        should_print=True,
    )

    print()
    print("[FULL TRAIN AFTER LEARNED MARKERS]")
    print("-" * 60)

    exact_count = 0
    total_score = 0
    results = []

    for idx, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        expected = pair["output"]

        print_scene_facts(f"pair {idx}", input_grid)

        predicted, explanation, applied_markers = predict_with_learned_markers(
            input_grid=input_grid,
            marker_rule=marker_rule,
        )

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
            "applied_markers": applied_markers,
        })

        print()
        print(f"PAIR {idx}")
        print(f"exact          : {exact}")
        print(f"score          : {score}")
        print(f"applied_markers: {applied_markers}")

        if not exact:
            print_diff_summary(
                f"FULL TRAIN PAIR {idx}",
                predicted,
                expected,
            )

        if SHOW_POPUPS:
            popup_grids(
                f"FULL TRAIN PAIR {idx} | exact={exact}",
                [
                    ("INPUT", input_grid),
                    ("EXPECTED", expected),
                    ("PREDICTED", predicted),
                ],
            )

    print()
    print("[FULL TRAIN AFTER LEARNED MARKERS SUMMARY]")
    print(f"Exact count: {exact_count}")
    print(f"Pair count : {len(train_pairs)}")
    print(f"Total score: {total_score}")

    return {
        "marker_rule": marker_rule,
        "exact_count": exact_count,
        "pair_count": len(train_pairs),
        "total_score": total_score,
        "results": results,
    }


# ============================================================
# FRAME-ONLY BASELINE CHECK
# ============================================================

def run_frame_only_baseline_check(train_pairs):
    print()
    print("=" * 60)
    print("DIAGNOSTIC — FRAME-ONLY BASELINE")
    print("=" * 60)

    rule = discover_visual_symbolic_rule_v2_for_task(train_pairs)

    exact_count = 0
    total_score = 0
    results = []

    for idx, pair in enumerate(train_pairs):
        predicted = apply_visual_symbolic_rule_v2(
            rule,
            pair["input"],
        )

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

    # Show one useful failure popup only, not every frame-only miss.
    for result in results:
        if not result["exact"]:
            idx = result["pair_index"]

            if PRINT_DIFFS:
                print_diff_summary(
                    f"FRAME-ONLY PAIR {idx}",
                    result["predicted"],
                    result["expected"],
                )

            popup_grids(
                f"FRAME-ONLY PAIR {idx}",
                [
                    ("INPUT", train_pairs[idx]["input"]),
                    ("EXPECTED", result["expected"]),
                    ("PREDICTED", result["predicted"]),
                ],
            )
            break

    return {
        "exact_count": exact_count,
        "pair_count": len(train_pairs),
        "total_score": total_score,
        "results": results,
    }


# ============================================================
# TEST PREDICTIONS — MAIN COMPETITION-STYLE FLOW
# ============================================================

def run_test_predictions(task, train_pairs, test_pairs, marker_rule):
    print()
    print("=" * 60)
    print("MAIN RESULT — TEST PREDICTIONS")
    print("=" * 60)
    print("Using anchor-compass first, then learned marker fallback.")

    results = []

    for idx, pair in enumerate(test_pairs):
        input_grid = pair["input"]

        print_scene_facts(f"TEST {idx}", input_grid)

        anchor_result = try_anchor_compass_merge_for_test_pair(
            task=task,
            test_pair=pair,
            test_index=idx,
            debug=True,
        )

        if anchor_result is not None:
            predicted = anchor_result["predicted"]
            explanation = None
            applied_markers = []
            strategy = anchor_result["strategy"]

        else:
            predicted, explanation, applied_markers = predict_with_learned_markers(
                input_grid=input_grid,
                marker_rule=marker_rule,
            )
            strategy = "visual_symbolic_ruleV2_learned_markers"

        print(f"test_strategy     : {strategy}")
        print(f"applied_markers   : {applied_markers}")

        if applied_markers:
            print_marker_decision_report(
                title=f"TEST DECISION REPORT {idx}",
                applied_markers=applied_markers,
            )

        if explanation is not None:
            print_structural_explanation(
                "V2 TEST STRUCTURAL EXPLANATION",
                explanation,
            )

        print()
        print("-" * 60)
        print(f"TEST PAIR {idx}")
        print("-" * 60)

        panels = [
            ("INPUT", input_grid),
            ("PREDICTED", predicted),
        ]

        if "output" in pair:
            expected = pair["output"]
            score = score_prediction(predicted, expected)
            exact = predicted == expected

            print(f"exact: {exact}")
            print(f"score: {score}")

            print_diff_summary(
                f"TEST PAIR {idx}",
                predicted,
                expected,
            )

            panels = [
                ("INPUT", input_grid),
                ("EXPECTED", expected),
                ("PREDICTED", predicted),
            ]

            results.append({
                "pair_index": idx,
                "strategy": strategy,
                "exact": exact,
                "score": score,
                "predicted": predicted,
                "expected": expected,
                "applied_markers": applied_markers,
            })

        else:
            results.append({
                "pair_index": idx,
                "strategy": strategy,
                "predicted": predicted,
                "applied_markers": applied_markers,
            })

        popup_grids(
            f"TEST PAIR {idx} | strategy={strategy}",
            panels,
        )

    return {
        "results": results,
    }


# ============================================================
# OPTIONAL DIAGNOSTIC — LEAVE ONE OUT CHECK
# ============================================================

def run_leave_one_out_check(train_pairs):
    print()
    print("=" * 60)
    print("OPTIONAL DIAGNOSTIC — LEAVE-ONE-OUT CHECK")
    print("=" * 60)
    print("This does NOT control the final learned rule.")
    print("This only tests whether the rule can still be learned when one train pair is hidden.")

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

        marker_rule = learn_marker_rule_from_train(
            train_pairs=visible_pairs,
            should_print=True,
        )

        predicted, explanation, applied_markers = predict_with_learned_markers(
            input_grid=hidden_pair["input"],
            marker_rule=marker_rule,
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
            "applied_markers": applied_markers,
        })

        print(f"applied_markers: {applied_markers}")

        if PRINT_DIFFS:
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
    print("[OPTIONAL DIAGNOSTIC — LEAVE ONE OUT]")
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
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("RUN_ONCE_V3 — VISUAL SYMBOLIC RULE V2")
    print("=" * 60)

    task_path = TARGET_TASK_FILE

    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Task file not found: {task_path}")

    task = load_task(task_path)

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    print(f"Task file  : {task_path}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Test pairs : {len(test_pairs)}")

    learned_full_result = None
    frame_only_result = None
    test_result = None
    loo_result = None

    marker_rule = None

    # ------------------------------------------------------------
    # 1. Main learned-marker full-train check.
    # ------------------------------------------------------------
    if RUN_FULL_TRAIN:
        learned_full_result = run_full_train_learned_marker_check(train_pairs)
        marker_rule = learned_full_result["marker_rule"]
    else:
        marker_rule = learn_marker_rule_from_train(
            train_pairs=train_pairs,
            should_print=True,
        )

    # ------------------------------------------------------------
    # 2. Frame-only baseline diagnostic.
    # ------------------------------------------------------------
    if RUN_FRAME_ONLY_BASELINE:
        frame_only_result = run_frame_only_baseline_check(train_pairs)

    # ------------------------------------------------------------
    # 3. Main competition-style test predictions.
    # ------------------------------------------------------------
    if RUN_TEST_PREDICTIONS:
        test_result = run_test_predictions(
            task=task,
            train_pairs=train_pairs,
            test_pairs=test_pairs,
            marker_rule=marker_rule,
        )

    # ------------------------------------------------------------
    # 4. LOO diagnostic LAST.
    # ------------------------------------------------------------
    if RUN_LEAVE_ONE_OUT:
        loo_result = run_leave_one_out_check(train_pairs)

    # ------------------------------------------------------------
    # Final summary.
    # ------------------------------------------------------------
    print()
    print("=" * 60)
    print("FINAL QUICK SUMMARY")
    print("=" * 60)

    print("MAIN RESULT")

    if learned_full_result is not None:
        print(
            "  Learned-marker full train : "
            f"{learned_full_result['exact_count']} / {learned_full_result['pair_count']}"
        )
    else:
        print("  Learned-marker full train : not run")

    if test_result is not None:
        print("  Test predictions          : see [MAIN RESULT — VISUAL SYMBOLIC V2 TEST PREDICTIONS]")
    else:
        print("  Test predictions          : not run")

    print()
    print("DIAGNOSTIC ONLY")

    if frame_only_result is not None:
        print(
            "  Frame-only full train     : "
            f"{frame_only_result['exact_count']} / {frame_only_result['pair_count']}"
        )
    else:
        print("  Frame-only full train     : not run")

    if loo_result is not None:
        print(
            "  Leave-one-out             : "
            f"{loo_result['exact_count']} / {loo_result['pair_count']}"
        )
    else:
        print("  Leave-one-out             : not run")

    print()
    print("Router                    : skipped")


if __name__ == "__main__":
    main()