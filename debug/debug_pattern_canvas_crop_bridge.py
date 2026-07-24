# debug_pattern_canvas_crop_bridge.py
import contextlib
import io
import json
import os

from core.grid_utils import print_grid
from OLD.arc_visualizer import show_three_grids

from reasoning.task_router import (
    FAMILY_SCORERS,
    FAMILY_APPLIERS,
    score_prediction,
)


MASK_COLOR = 8
FAMILY_NAME = "pattern_canvas_family"
QUIET_ENGINE_OUTPUT = True


def quiet_call(fn, *args, quiet=True, **kwargs):
    if not quiet:
        return fn(*args, **kwargs)

    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        return fn(*args, **kwargs)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def unwrap_task(raw, task_id):
    if "train" in raw:
        return raw

    if task_id in raw:
        return raw[task_id]

    if isinstance(raw, dict) and len(raw) == 1:
        return raw[next(iter(raw))]

    raise KeyError(f"Could not find task {task_id}")


def load_task(task_id_or_path):
    base_dir = os.path.dirname(__file__)
    value = task_id_or_path.strip().strip('"')

    if value.endswith(".json") and os.path.exists(value):
        raw = load_json(value)
        task_id = os.path.splitext(os.path.basename(value))[0]
        return task_id, unwrap_task(raw, task_id)

    failure_path = os.path.join(
        base_dir,
        "data_failures",
        "extracted_tasks",
        value + ".json",
    )

    if os.path.exists(failure_path):
        raw = load_json(failure_path)
        return value, unwrap_task(raw, value)

    data_path = os.path.join(base_dir, "data", "data.json")

    if os.path.exists(data_path):
        data = load_json(data_path)

        if value in data:
            return value, data[value]

    raise FileNotFoundError(value)


def find_family_scorer(family_name):
    wanted_name = "score_" + family_name

    for item in FAMILY_SCORERS:
        if callable(item):
            if item.__name__ == wanted_name:
                return item

        if isinstance(item, dict):
            name = item.get("name") or item.get("family")
            scorer = item.get("scorer") or item.get("fn") or item.get("function")

            if name == family_name:
                return scorer

        elif isinstance(item, tuple) and len(item) >= 2:
            name = item[0]
            scorer = item[1]

            if name == family_name:
                return scorer

    return None


def find_family_applier(family_name):
    wanted_name = "apply_" + family_name

    if isinstance(FAMILY_APPLIERS, dict):
        return FAMILY_APPLIERS.get(family_name)

    for item in FAMILY_APPLIERS:
        if callable(item):
            if item.__name__ == wanted_name:
                return item

        if isinstance(item, dict):
            name = item.get("name") or item.get("family")
            applier = item.get("applier") or item.get("fn") or item.get("function")

            if name == family_name:
                return applier

        elif isinstance(item, tuple) and len(item) >= 2:
            name = item[0]
            applier = item[1]

            if name == family_name:
                return applier

    return None


def find_mask_bbox(grid, mask_color=MASK_COLOR):
    cells = []

    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == mask_color:
                cells.append((r, c))

    if not cells:
        return None

    top = min(r for r, _ in cells)
    bottom = max(r for r, _ in cells)
    left = min(c for _, c in cells)
    right = max(c for _, c in cells)

    return {
        "top": top,
        "left": left,
        "bottom": bottom,
        "right": right,
        "height": bottom - top + 1,
        "width": right - left + 1,
    }


def crop_bbox(grid, bbox):
    if grid is None or bbox is None:
        return None

    grid_h = len(grid)
    grid_w = len(grid[0]) if grid_h else 0

    if bbox["bottom"] >= grid_h:
        return None

    if bbox["right"] >= grid_w:
        return None

    return [
        row[bbox["left"]:bbox["right"] + 1]
        for row in grid[bbox["top"]:bbox["bottom"] + 1]
    ]


def grid_shape(grid):
    if grid is None:
        return "None"

    h = len(grid)
    w = len(grid[0]) if h else 0
    return f"{h}x{w}"


def extract_rule_from_scorer_result(result):
    """
    The router scorer may return:
        dict
        tuple(result_dict, rule_dict, extra)
        None

    We want the actual task rule if it exists.
    """
    if result is None:
        return None

    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, dict) and item.get("task_rule") is not None:
                return item["task_rule"]

        for item in result:
            if isinstance(item, dict) and item.get("rule") is not None:
                return item["rule"]

        for item in result:
            if isinstance(item, dict) and item.get("rule_type") is not None:
                return item

        return None

    if isinstance(result, dict):
        if result.get("task_rule") is not None:
            return result["task_rule"]

        if result.get("rule") is not None:
            return result["rule"]

        if result.get("rule_type") is not None:
            return result

    return None


def get_pattern_canvas_rule(train_pairs):
    scorer = find_family_scorer(FAMILY_NAME)

    if scorer is None:
        print(f"Missing scorer for {FAMILY_NAME}")
        print(f"FAMILY_SCORERS type: {type(FAMILY_SCORERS)}")
        print(f"FAMILY_SCORERS value: {FAMILY_SCORERS}")
        return None

    result = quiet_call(
        scorer,
        train_pairs,
        quiet=QUIET_ENGINE_OUTPUT,
    )

    print()
    print("RAW PATTERN CANVAS SCORE RESULT")
    print("-" * 80)
    print(result)

    return extract_rule_from_scorer_result(result)


def apply_pattern_canvas_rule(rule, input_grid, expected_grid=None, pair_index=None):
    applier = find_family_applier(FAMILY_NAME)

    if applier is None:
        print(f"Missing applier for {FAMILY_NAME}")
        print(f"FAMILY_APPLIERS type: {type(FAMILY_APPLIERS)}")
        print(f"FAMILY_APPLIERS value: {FAMILY_APPLIERS}")
        return None

    call_attempts = [
        {
            "task_rule": rule,
            "input_grid": input_grid,
            "expected_grid": expected_grid,
            "pair_index": pair_index,
        },
        {
            "rule": rule,
            "input_grid": input_grid,
            "expected_grid": expected_grid,
            "pair_index": pair_index,
        },
    ]

    for kwargs in call_attempts:
        try:
            return quiet_call(
                applier,
                quiet=QUIET_ENGINE_OUTPUT,
                **kwargs,
            )
        except TypeError:
            pass

    try:
        return quiet_call(
            applier,
            rule,
            input_grid,
            expected_grid,
            pair_index,
            quiet=QUIET_ENGINE_OUTPUT,
        )
    except TypeError:
        pass

    try:
        return quiet_call(
            applier,
            rule,
            input_grid,
            quiet=QUIET_ENGINE_OUTPUT,
        )
    except TypeError:
        return None


def get_stored_train_prediction(rule, pair_index):
    """
    If pattern_canvas_family only stored train predictions inside inner_score,
    pull them out so we can still inspect what it thought.
    """
    if not isinstance(rule, dict):
        return None

    inner_score = rule.get("inner_score")

    if not isinstance(inner_score, dict):
        return None

    results = inner_score.get("results", [])

    for result in results:
        if result.get("pair_index") == pair_index:
            return result.get("predicted")

    return None


def main():
    task_id_or_path = input("Task id or json path: ").strip()
    task_id, task = load_task(task_id_or_path)

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    print()
    print("=" * 80)
    print(f"PATTERN CANVAS CROP BRIDGE: {task_id}")
    print("=" * 80)

    rule = get_pattern_canvas_rule(train_pairs)

    print()
    print("PATTERN CANVAS RULE")
    print("-" * 80)

    if isinstance(rule, dict):
        print(f"Rule keys : {sorted(rule.keys())}")
        print(f"Family    : {rule.get('family')}")
        print(f"Rule type : {rule.get('rule_type')}")
        print(f"Inner     : {rule.get('chosen_inner_strategy')}")
    else:
        print(f"Rule type : {type(rule)}")

    if rule is None:
        print("No pattern_canvas rule found.")
        return

    print()
    print("=" * 80)
    print("TRAIN CHECK: RAW VS CROPPED")
    print("=" * 80)

    train_exact_raw = 0
    train_exact_cropped = 0

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        expected_grid = pair["output"]

        raw_prediction = apply_pattern_canvas_rule(
            rule,
            input_grid,
            expected_grid=expected_grid,
            pair_index=pair_index,
        )

        if raw_prediction is None:
            raw_prediction = get_stored_train_prediction(rule, pair_index)

        bbox = find_mask_bbox(input_grid)
        cropped_prediction = crop_bbox(raw_prediction, bbox)

        raw_exact = raw_prediction == expected_grid
        cropped_exact = cropped_prediction == expected_grid

        if raw_exact:
            train_exact_raw += 1

        if cropped_exact:
            train_exact_cropped += 1

        raw_score = (
            score_prediction(raw_prediction, expected_grid)
            if raw_prediction is not None
            else 0
        )

        cropped_score = (
            score_prediction(cropped_prediction, expected_grid)
            if cropped_prediction is not None
            else 0
        )

        print()
        print("-" * 80)
        print(f"TRAIN PAIR {pair_index + 1}")
        print(f"mask bbox      : {bbox}")
        print(f"raw shape      : {grid_shape(raw_prediction)}")
        print(f"cropped shape  : {grid_shape(cropped_prediction)}")
        print(f"expected shape : {grid_shape(expected_grid)}")
        print(f"raw exact      : {raw_exact}")
        print(f"raw score      : {raw_score}")
        print(f"cropped exact  : {cropped_exact}")
        print(f"cropped score  : {cropped_score}")

        print_grid(expected_grid, "EXPECTED")

        if raw_prediction is not None:
            print_grid(raw_prediction, "RAW PATTERN PREDICTION")
        else:
            print("RAW PATTERN PREDICTION: None")

        if cropped_prediction is not None:
            print_grid(cropped_prediction, "CROPPED PATTERN PREDICTION")

            show_three_grids(
                input_grid,
                expected_grid,
                cropped_prediction,
                title_a=f"{task_id} TRAIN {pair_index + 1} INPUT",
                title_b="EXPECTED",
                title_c="CROPPED PATTERN PREDICTED",
            )
        else:
            print("CROPPED PATTERN PREDICTION: None")

    print()
    print("=" * 80)
    print("TRAIN SUMMARY")
    print("=" * 80)
    print(f"Raw exact     : {train_exact_raw}/{len(train_pairs)}")
    print(f"Cropped exact : {train_exact_cropped}/{len(train_pairs)}")

    print()
    print("=" * 80)
    print("TEST CHECK")
    print("=" * 80)

    for test_index, pair in enumerate(test_pairs):
        input_grid = pair["input"]

        raw_prediction = apply_pattern_canvas_rule(
            rule,
            input_grid,
            expected_grid=None,
            pair_index=None,
        )

        bbox = find_mask_bbox(input_grid)
        cropped_prediction = crop_bbox(raw_prediction, bbox)

        print()
        print("-" * 80)
        print(f"TEST PAIR {test_index + 1}")
        print(f"mask bbox     : {bbox}")
        print(f"raw shape     : {grid_shape(raw_prediction)}")
        print(f"cropped shape : {grid_shape(cropped_prediction)}")

        print_grid(input_grid, "TEST INPUT")

        if raw_prediction is not None:
            print_grid(raw_prediction, "RAW PATTERN TEST PREDICTION")
        else:
            print("RAW PATTERN TEST PREDICTION: None")

        if cropped_prediction is not None:
            print_grid(cropped_prediction, "CROPPED TEST PREDICTION")

            show_three_grids(
                input_grid,
                cropped_prediction,
                cropped_prediction,
                title_a=f"{task_id} TEST {test_index + 1} INPUT",
                title_b="CROPPED PREDICTED",
                title_c="CROPPED PREDICTED",
            )
        else:
            print("CROPPED TEST PREDICTION: None")


if __name__ == "__main__":
    main()