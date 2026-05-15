# vision/debug_visual_abstraction_mapper.py
"""
Debug Visual Abstraction Mapper

Purpose:
    Load one ARC task.
    Learn the best visual abstraction view.
    Learn input abstraction -> output abstraction/template mappings.
    Match each test input to the closest learned train mapping.
    Print predictions.
    Show popup windows so we can visually compare grids.

Important:
    This file owns debugging:
        - console prints
        - popup windows
        - adapter action logs
        - actual-solve comparison

    reasoning/visual_abstraction_mapper.py should stay clean and only return data.

Popup visual order:
    TEST INPUT -> ACTUAL SOLVE -> OUR PREDICTED

Current target task:
    data_failures/extracted_tasks/2d0172a1.json

Run:
    python debug_visual_abstraction_mapper.py
"""

import json
import os
import sys
import tkinter as tk


# ============================================================
# Make project imports work when running this file directly
# ============================================================

CURRENT_FILE = os.path.abspath(__file__)
VISION_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(VISION_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Imports
# ============================================================

from reasoning.visual_abstraction_rule_learner import (
    discover_visual_abstraction_rule_for_task,
)

from reasoning.visual_abstraction_mapper import (
    learn_visual_abstraction_mapping,
    choose_closest_mapping_for_input,
    create_prediction_from_closest_mapping,
    create_adapted_prediction_from_closest_mapping,
    learn_assignment_marker_growth_rules,
    learn_generic_marker_growth_rule,
)
from reasoning.archive.ring_blob_output_builder import (
    learn_ring_blob_output_builder,
    predict_with_ring_blob_output_builder,
    print_builder_training_summary,
    print_builder_prediction_summary,
)
from reasoning.archive.ring_blob_rule_synthesizer import (
    learn_ring_blob_rule_synthesizer,
    predict_with_ring_blob_rule_synthesizer,
    print_rule_synthesizer_summary,
    print_rule_synthesizer_prediction_summary,
)

from reasoning.ring_detector import print_ring_summary
from reasoning.archive.ring_blob_scene import print_ring_blob_scene_summary


# ============================================================
# Config
# ============================================================

TASK_ID = "2d0172a1"

TASK_PATH = os.path.join(
    PROJECT_ROOT,
    "data_failures",
    "extracted_tasks",
    f"{TASK_ID}.json",
)

SHOW_POPUPS = True
CELL_SIZE = 28


# ============================================================
# Actual solved outputs for comparison
# ============================================================

ACTUAL_SOLVES = {
    1: [
        [9, 9, 9, 9, 9, 9, 9, 9, 9, 7, 7, 7],
        [9, 7, 7, 7, 7, 7, 7, 7, 9, 7, 7, 7],
        [9, 7, 9, 9, 9, 9, 9, 7, 9, 7, 7, 7],
        [9, 7, 9, 7, 7, 7, 9, 7, 9, 7, 7, 7],
        [9, 7, 9, 7, 9, 7, 9, 7, 9, 7, 9, 7],
        [9, 7, 9, 7, 7, 7, 9, 7, 9, 7, 7, 7],
        [9, 7, 9, 9, 9, 9, 9, 7, 9, 7, 7, 7],
        [9, 7, 7, 7, 7, 7, 7, 7, 9, 7, 7, 7],
        [9, 7, 7, 7, 9, 7, 7, 7, 9, 7, 7, 7],
        [9, 7, 7, 7, 7, 7, 7, 7, 9, 7, 7, 7],
        [9, 9, 9, 9, 9, 9, 9, 9, 9, 7, 7, 7],
    ],
    2: [
        [6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8],
        [6, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8],
        [6, 8, 6, 6, 6, 6, 6, 8, 6, 8, 8, 8],
        [6, 8, 6, 8, 8, 8, 6, 8, 6, 8, 8, 8],
        [6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8],
        [6, 8, 6, 8, 8, 8, 6, 8, 6, 8, 8, 8],
        [6, 8, 6, 6, 6, 6, 6, 8, 6, 8, 8, 8],
        [6, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8],
        [6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8],
    ],
}


# ============================================================
# Basic task/loading helpers
# ============================================================

def load_task(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def unwrap_task(raw):
    """
    Find the real ARC task object.

    Handles:
        {"train": [...], "test": [...]}

    and:
        {"2d0172a1": {"train": [...], "test": [...]}}

    and common wrappers:
        {"task": {...}}
        {"data": {...}}
    """
    if not isinstance(raw, dict):
        return raw

    if "train" in raw and "test" in raw:
        return raw

    if len(raw) == 1:
        only_value = next(iter(raw.values()))

        if isinstance(only_value, dict):
            if "train" in only_value and "test" in only_value:
                return only_value

    for key in ["task", "data", "arc_task", "raw_task"]:
        value = raw.get(key)

        if isinstance(value, dict):
            if "train" in value and "test" in value:
                return value

    for value in raw.values():
        if isinstance(value, dict):
            if "train" in value and "test" in value:
                return value

    return raw


def get_train_pairs(raw_task):
    task = unwrap_task(raw_task)

    if not isinstance(task, dict):
        return []

    train = task.get("train", [])

    if train:
        return train

    for key in ["train_pairs", "training", "examples"]:
        value = task.get(key)

        if isinstance(value, list):
            return value

    return []


def get_test_pairs(raw_task):
    task = unwrap_task(raw_task)

    if not isinstance(task, dict):
        return []

    test = task.get("test", [])

    if test:
        return test

    for key in ["test_pairs", "testing", "tests"]:
        value = task.get(key)

        if isinstance(value, list):
            return value

    return []


# ============================================================
# Grid helpers
# ============================================================

def grid_shape(grid):
    if not grid:
        return (0, 0)

    return (len(grid), len(grid[0]))


def grids_equal(a, b):
    return a == b


def count_wrong_cells(predicted_grid, actual_grid):
    """
    Count wrong cells between prediction and actual solve.

    Dimension differences count as wrong cells too.
    """
    if predicted_grid is None or actual_grid is None:
        return None

    pred_h, pred_w = grid_shape(predicted_grid)
    actual_h, actual_w = grid_shape(actual_grid)

    h = max(pred_h, actual_h)
    w = max(pred_w, actual_w)

    wrong_count = 0

    for r in range(h):
        for c in range(w):
            pred_value = None
            actual_value = None

            if r < pred_h and c < pred_w:
                pred_value = predicted_grid[r][c]

            if r < actual_h and c < actual_w:
                actual_value = actual_grid[r][c]

            if pred_value != actual_value:
                wrong_count += 1

    return wrong_count


def print_grid(title, grid):
    print()
    print(title)
    print("-" * 60)

    if grid is None:
        print("None")
        return

    h, w = grid_shape(grid)
    print(f"shape: ({h}, {w})")

    for row in grid:
        print(" ".join(str(x) for x in row))


def print_pair_counts(train_pairs, test_pairs):
    print()
    print("TASK PAIR COUNTS")
    print("-" * 60)
    print(f"train pairs: {len(train_pairs)}")
    print(f"test pairs : {len(test_pairs)}")


# ============================================================
# Debug print helpers
# ============================================================

def print_feature_delta(feature_delta):
    """
    Print test-vs-matched-train abstraction differences.
    """
    print()
    print("TEST vs MATCHED TRAIN FEATURE DELTA")
    print("-" * 60)

    print(f"ring_count_delta        : {feature_delta.get('ring_count_delta')}")
    print(f"blob_count_delta        : {feature_delta.get('blob_count_delta')}")
    print(f"inside_blob_count_delta : {feature_delta.get('inside_blob_count_delta')}")
    print(f"outside_blob_count_delta: {feature_delta.get('outside_blob_count_delta')}")

    print(f"matched train signature : {feature_delta.get('matched_train_assignment_signature')}")
    print(f"test signature          : {feature_delta.get('test_assignment_signature')}")

    print("assignment delta:")

    for key, value in feature_delta.get("assignment_delta", {}).items():
        print(f"  {key}: {value}")


def print_assignment_marker_growth_rules(mapping_rule):
    """
    Print learned assignment-level marker growth rules.
    """
    rules = mapping_rule.get("assignment_marker_growth_rules", {})

    print()
    print("ASSIGNMENT MARKER GROWTH RULES")
    print("-" * 60)

    if not rules:
        print("No assignment growth rules learned.")
        return

    for assignment_label, rule in rules.items():
        print(f"{assignment_label}:")
        print(f"  source pair    : {rule.get('source_pair')}")
        print(f"  direction      : {rule.get('direction')}")
        print(f"  step           : {rule.get('step')}")
        print(f"  learned centers: {rule.get('learned_centers')}")
        print(f"  confidence     : {rule.get('confidence')}")


def print_generic_marker_growth_rule(mapping_rule):
    """
    Print the generic task-level marker growth rule.
    """
    rule = mapping_rule.get("generic_marker_growth_rule")

    print()
    print("GENERIC MARKER GROWTH RULE")
    print("-" * 60)

    if rule is None:
        print("No generic marker growth rule learned.")
        return

    print(f"source assignment: {rule.get('source_assignment')}")
    print(f"source pair      : {rule.get('source_pair')}")
    print(f"direction        : {rule.get('direction')}")
    print(f"step             : {rule.get('step')}")
    print(f"learned centers  : {rule.get('learned_centers')}")
    print(f"confidence       : {rule.get('confidence')}")


def print_adapter_actions(actions):
    """
    Print what the adapter did.
    """
    print()
    print("ADAPTER ACTION LOG")
    print("-" * 60)

    if not actions:
        print("No adapter actions.")
        return

    for index, action in enumerate(actions, start=1):
        print(f"Action {index}")
        print(f"  assignment   : {action.get('assignment')}")
        print(f"  success      : {action.get('success')}")
        print(f"  reason       : {action.get('reason')}")
        print(f"  placed at    : {action.get('placed_at')}")
        print(f"  marker color : {action.get('marker_color')}")
        print(f"  pattern step : {action.get('pattern_step')}")
        print(f"  used generic : {action.get('used_generic_growth_rule')}")

        growth_rule = action.get("growth_rule_used")

        if growth_rule is None:
            print("  growth rule  : None")
        else:
            print(f"  growth rule  : {growth_rule.get('assignment')}")
            print(f"    direction  : {growth_rule.get('direction')}")
            print(f"    step       : {growth_rule.get('step')}")
            print(f"    source pair: {growth_rule.get('source_pair')}")

        anchor_group = action.get("anchor_group")

        if anchor_group is None:
            print("  anchor group : None")
        else:
            print(f"  anchor group bbox  : {anchor_group.get('bbox')}")
            print(f"  anchor group cells : {anchor_group.get('cell_count')}")
            print(f"  anchor group colors: {anchor_group.get('colors')}")

        print()


def print_visual_abstraction_mapping(mapping_rule):
    """
    Print the learned input/output mapping summary.
    """
    print()
    print("VISUAL ABSTRACTION MAPPING")
    print("=" * 60)
    print(f"Preferred view type : {mapping_rule.get('preferred_view_type')}")
    print(f"Pair count          : {mapping_rule.get('pair_count')}")

    patterns = mapping_rule.get("known_patterns", {})

    print()
    print("Known task-level patterns")
    print("-" * 60)
    print(f"Output shapes       : {patterns.get('output_shapes')}")
    print(f"Frame depths        : {patterns.get('frame_depths')}")
    print(f"Ring counts         : {patterns.get('ring_counts')}")
    print(f"Blob counts         : {patterns.get('blob_counts')}")
    print(f"Marker counts       : {patterns.get('marker_counts')}")
    print(f"Marker group counts : {patterns.get('marker_group_counts')}")

    print("Assignment signatures:")

    for sig in patterns.get("assignment_signatures", []):
        print(f"  {sig}")

    print()
    print("Pair mappings")
    print("=" * 60)

    for mapping in mapping_rule.get("pair_mappings", []):
        pair_index = mapping["pair_index"]
        input_features = mapping["input_features"]
        output_abs = mapping["output_abstraction"]
        output_template = mapping["output_template"]
        relation = mapping["learned_relation"]

        print()
        print(f"TRAIN PAIR {pair_index}")
        print("-" * 60)

        print("Input abstraction")
        print(f"  rings               : {input_features.get('ring_count')}")
        print(f"  blobs               : {input_features.get('blob_count')}")
        print(f"  inside blobs         : {input_features.get('inside_blob_count')}")
        print(f"  outside blobs        : {input_features.get('outside_blob_count')}")
        print(f"  assignment signature : {input_features.get('assignment_signature')}")

        print("Output abstraction")
        print(f"  shape               : {output_abs.get('shape')}")
        print(f"  border color         : {output_abs.get('border_color')}")
        print(f"  fill color           : {output_abs.get('fill_color')}")
        print(f"  border ratio         : {output_abs.get('border_ratio')}")
        print(f"  frame depth          : {output_abs.get('frame_depth')}")
        print(f"  interior frame cells : {output_abs.get('interior_frame_cells')}")
        print(f"  marker groups        : {output_abs.get('marker_group_count')}")

        print("Learned output template")
        print(f"  marker count         : {output_template.get('marker_count')}")
        print(f"  marker group count   : {output_template.get('marker_group_count')}")

        marker_groups = output_template.get("marker_groups", [])

        for group in marker_groups:
            print(
                "    group "
                f"{group.get('group_index')}: "
                f"cells={group.get('cell_count')}, "
                f"bbox={group.get('bbox')}, "
                f"colors={group.get('colors')}"
            )

        print("Learned relation")
        print(
            "  input rings -> output frame depth: "
            f"{relation['rings_to_frame_depth'][0]} -> "
            f"{relation['rings_to_frame_depth'][1]}"
        )
        print(
            "  input blobs -> marker cells      : "
            f"{relation['blobs_to_marker_cells'][0]} -> "
            f"{relation['blobs_to_marker_cells'][1]}"
        )
        print(
            "  input blobs -> marker groups     : "
            f"{relation['blobs_to_marker_groups'][0]} -> "
            f"{relation['blobs_to_marker_groups'][1]}"
        )


def print_test_match(match):
    """
    Print closest train mapping chosen for a test input.
    """
    if match is None:
        print("No match found.")
        return

    output_abs = match["matched_output_abstraction"]
    output_template = match["matched_output_template"]

    print(f"test signature      : {match['test_signature']}")
    print(f"matched train pair  : {match['matched_train_pair']}")
    print(f"match score         : {match['match_score']}")
    print(f"matched output shape: {output_abs.get('shape')}")
    print(f"matched frame depth : {output_abs.get('frame_depth')}")
    print(f"matched markers     : {output_template.get('marker_count')}")
    print(f"matched groups      : {output_template.get('marker_group_count')}")


def print_actual_solve_check(predicted, actual_solve):
    """
    Print direct comparison between our prediction and actual solve.
    """
    print()
    print("ACTUAL SOLVE CHECK")
    print("-" * 60)

    if actual_solve is None:
        print("No actual solve available for this test pair.")
        return

    predicted_shape = grid_shape(predicted)
    actual_shape = grid_shape(actual_solve)
    exact = grids_equal(predicted, actual_solve)
    wrong_count = count_wrong_cells(predicted, actual_solve)

    print(f"predicted shape : {predicted_shape}")
    print(f"actual shape    : {actual_shape}")
    print(f"exact match     : {exact}")
    print(f"wrong cell count: {wrong_count}")


# ============================================================
# Popup visualizer helpers
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
    8: "#7FDBFF",  # cyan
    9: "#870C25",  # dark red
}


def draw_grid_on_canvas(canvas, grid, x_offset, y_offset, cell_size=28):
    """
    Draw a grid as color blocks only.

    Important:
        No numbers are drawn inside the cells.
        This lets us see the ARC image the way a human sees it.
    """
    if grid is None:
        return

    h, w = grid_shape(grid)

    for r in range(h):
        for c in range(w):
            value = grid[r][c]
            color = ARC_COLORS.get(value, "#FFFFFF")

            x1 = x_offset + c * cell_size
            y1 = y_offset + r * cell_size
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


def show_grids_popup(title, labeled_grids, cell_size=28):
    """
    Show grids side by side.

    Close the popup to continue to the next pair.
    """
    visible_items = [
        (label, grid)
        for label, grid in labeled_grids
        if grid is not None
    ]

    if not visible_items:
        return

    padding = 30
    label_height = 35
    gap = 50

    widths = []
    heights = []

    for label, grid in visible_items:
        h, w = grid_shape(grid)
        widths.append(w * cell_size)
        heights.append(h * cell_size)

    canvas_width = padding * 2 + sum(widths) + gap * (len(widths) - 1)
    canvas_height = padding * 2 + label_height + max(heights)

    print()
    print(f"OPENING POPUP: {title}")
    print("Close the popup window to continue.")

    root = tk.Tk()
    root.title(title)

    canvas = tk.Canvas(
        root,
        width=canvas_width,
        height=canvas_height,
        bg="white",
    )
    canvas.pack()

    x = padding

    for idx, (label, grid) in enumerate(visible_items):
        canvas.create_text(
            x + widths[idx] / 2,
            padding / 2,
            text=label,
            fill="black",
            font=("Arial", 12, "bold"),
        )

        draw_grid_on_canvas(
            canvas,
            grid,
            x_offset=x,
            y_offset=padding + label_height,
            cell_size=cell_size,
        )

        x += widths[idx] + gap

    root.mainloop()


# ============================================================
# Main debug flow
# ============================================================

def main():
    if not os.path.exists(TASK_PATH):
        raise FileNotFoundError(f"Task file not found: {TASK_PATH}")

    raw_task = load_task(TASK_PATH)

    print(f"Loaded task from: {TASK_PATH}")
    print("=" * 60)
    print(f"DEBUG VISUAL ABSTRACTION MAPPER — TASK {TASK_ID}")
    print("=" * 60)

    if isinstance(raw_task, dict):
        print(f"Top-level JSON keys: {list(raw_task.keys())}")
    else:
        print(f"Top-level JSON type: {type(raw_task)}")

    task = unwrap_task(raw_task)
    train_pairs = get_train_pairs(task)
    test_pairs = get_test_pairs(task)

    print_pair_counts(train_pairs, test_pairs)

    if not train_pairs:
        print()
        print("No train pairs found.")
        return

    visual_rule = discover_visual_abstraction_rule_for_task(train_pairs)

    if visual_rule is None:
        print()
        print("No visual abstraction rule learned.")
        return

    best_view_type = visual_rule.get("best_view_type", "ring_blob_view")

    print()
    print(f"Best learned view type: {best_view_type}")

    mapping_rule = learn_visual_abstraction_mapping(
        train_pairs,
        preferred_view_type=best_view_type,
    )

    ring_blob_builder = learn_ring_blob_output_builder(train_pairs)
    print_builder_training_summary(ring_blob_builder)
    ring_blob_synthesizer = learn_ring_blob_rule_synthesizer(train_pairs)
    print_rule_synthesizer_summary(ring_blob_synthesizer)
    print_visual_abstraction_mapping(mapping_rule)

    learn_assignment_marker_growth_rules(mapping_rule)
    print_assignment_marker_growth_rules(mapping_rule)

    learn_generic_marker_growth_rule(mapping_rule)
    print_generic_marker_growth_rule(mapping_rule)

    if SHOW_POPUPS:
        print()
        print("TRAINING PAIR POPUPS")
        print("=" * 60)

        for train_index, train_pair in enumerate(train_pairs, start=1):
            train_input = train_pair.get("input")
            train_expected = train_pair.get("output")

            print()
            print(f"OPENING TRAIN POPUP: TRAIN PAIR {train_index}")
            print("Close the popup window to continue.")

            show_grids_popup(
                title=f"TASK {TASK_ID} — TRAIN PAIR {train_index}",
                labeled_grids=[
                    ("TRAIN INPUT", train_input),
                    ("TRAIN EXPECTED", train_expected),
                ],
                cell_size=CELL_SIZE,
            )

    print()
    print("TEST MATCHING")
    print("=" * 60)

    if not test_pairs:
        print("No test pairs found.")
        return

    for test_index, test_pair in enumerate(test_pairs, start=1):
        input_grid = test_pair.get("input")
        actual_solve = ACTUAL_SOLVES.get(test_index)

        print()
        print(f"TEST PAIR {test_index}")
        print("-" * 60)

        if input_grid is None:
            print("No input grid found for this test pair.")
            continue

        print_grid("TEST INPUT", input_grid)

        print_ring_summary(input_grid)
        print_ring_blob_scene_summary(input_grid)
        builder_result = predict_with_ring_blob_output_builder(
            ring_blob_builder,
            input_grid,
        )
        print_builder_prediction_summary(builder_result)

        builder_prediction = builder_result.get("prediction")

        synth_result = predict_with_ring_blob_rule_synthesizer(
            ring_blob_synthesizer,
            input_grid,
        )

        print_rule_synthesizer_prediction_summary(synth_result)

        synth_prediction = synth_result.get("prediction")
        if actual_solve is not None:
            print_grid("ACTUAL SOLVE", actual_solve)

        match = choose_closest_mapping_for_input(
            mapping_rule,
            input_grid,
        )

        if match is None:
            print("No closest mapping found.")
            continue

        print_test_match(match)
        print_feature_delta(match["feature_delta"])

        learned_template_prediction = create_prediction_from_closest_mapping(
            mapping_rule,
            input_grid,
        )

        print_grid(
            "LEARNED TEMPLATE PREDICTION",
            learned_template_prediction,
        )

        predicted, adapter_actions = create_adapted_prediction_from_closest_mapping(
            mapping_rule,
            input_grid,
            return_actions=True,
        )

        print_grid("OUR FINAL PREDICTED OUTPUT", predicted)
        print_adapter_actions(adapter_actions)
        print_actual_solve_check(predicted, actual_solve)

        if builder_prediction is not None:
            print_grid("SCENE-BASED BUILDER PREDICTION", builder_prediction)

        if synth_prediction is not None:
            print_grid("ALL-TRAIN SYNTHESIZER PREDICTION", synth_prediction)

        if synth_prediction is not None:
            print()
            print("ALL-TRAIN SYNTHESIZER ACTUAL SOLVE CHECK")
            print_actual_solve_check(synth_prediction, actual_solve)

        if SHOW_POPUPS:
            show_grids_popup(
                title=f"TASK {TASK_ID} — TEST PAIR {test_index}",
                labeled_grids=[
                    ("TEST INPUT", input_grid),
                    ("ACTUAL SOLVE", actual_solve),
                    ("OUR PREDICTED", predicted),
                ],
                cell_size=CELL_SIZE,
            )


if __name__ == "__main__":
    main()