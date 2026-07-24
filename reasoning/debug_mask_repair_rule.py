# debug_mask_repair_rule.py
import json
import os

from core.grid_utils import print_grid
from OLD.arc_visualizer import show_three_grids


MASK_COLOR = 8


# ============================================================
# LOAD TASK
# ============================================================

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
    base_dir = os.path.dirname(os.path.dirname(__file__))
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


# ============================================================
# MASK HELPERS
# ============================================================

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
        "cell_count": len(cells),
    }


def is_solid_bbox(grid, bbox, mask_color=MASK_COLOR):
    if bbox is None:
        return False

    for r in range(bbox["top"], bbox["bottom"] + 1):
        for c in range(bbox["left"], bbox["right"] + 1):
            if grid[r][c] != mask_color:
                return False

    return True


def get_value_or_none(grid, r, c):
    h = len(grid)
    w = len(grid[0]) if h else 0

    if r < 0 or c < 0 or r >= h or c >= w:
        return None

    value = grid[r][c]

    if value == MASK_COLOR:
        return None

    return value


# ============================================================
# CANDIDATE REPAIR METHODS
# ============================================================

def predict_from_transform(grid, bbox, transform_name):
    h = len(grid)
    w = len(grid[0]) if h else 0

    top = bbox["top"]
    left = bbox["left"]
    height = bbox["height"]
    width = bbox["width"]

    output = []

    for i in range(height):
        row = []

        for j in range(width):
            r = top + i
            c = left + j

            if transform_name == "vertical_mirror":
                rr, cc = r, w - 1 - c

            elif transform_name == "horizontal_mirror":
                rr, cc = h - 1 - r, c

            elif transform_name == "both_mirror":
                rr, cc = h - 1 - r, w - 1 - c

            elif transform_name == "main_diagonal":
                rr, cc = c, r

            elif transform_name == "anti_diagonal":
                rr, cc = w - 1 - c, h - 1 - r

            elif transform_name == "rotate_90_position":
                rr, cc = c, h - 1 - r

            elif transform_name == "rotate_270_position":
                rr, cc = w - 1 - c, r

            else:
                return None

            row.append(get_value_or_none(grid, rr, cc))

        output.append(row)

    if any(value is None for row in output for value in row):
        return None

    return output


def score_patch(predicted, expected):
    if predicted is None:
        return 0, False

    if len(predicted) != len(expected):
        return 0, False

    if predicted and expected and len(predicted[0]) != len(expected[0]):
        return 0, False

    score = 0

    for r in range(len(expected)):
        for c in range(len(expected[0])):
            if predicted[r][c] == expected[r][c]:
                score += 1

    exact = predicted == expected
    return score, exact


def discover_mask_repair_rule(train_pairs):
    candidate_names = [
        "vertical_mirror",
        "horizontal_mirror",
        "both_mirror",
        "main_diagonal",
        "anti_diagonal",
        "rotate_90_position",
        "rotate_270_position",
    ]

    best_rule = None
    best_exact_count = -1
    best_total_score = -1

    for candidate_name in candidate_names:
        exact_count = 0
        total_score = 0
        pair_results = []

        for pair_index, pair in enumerate(train_pairs):
            input_grid = pair["input"]
            expected_grid = pair["output"]

            bbox = find_mask_bbox(input_grid)

            if bbox is None or not is_solid_bbox(input_grid, bbox):
                pair_results.append({
                    "pair_index": pair_index,
                    "score": 0,
                    "exact": False,
                    "bbox": bbox,
                })
                continue

            predicted = predict_from_transform(
                input_grid,
                bbox,
                candidate_name,
            )

            score, exact = score_patch(predicted, expected_grid)

            if exact:
                exact_count += 1

            total_score += score

            pair_results.append({
                "pair_index": pair_index,
                "score": score,
                "exact": exact,
                "bbox": bbox,
            })

        if (
            exact_count > best_exact_count
            or (
                exact_count == best_exact_count
                and total_score > best_total_score
            )
        ):
            best_exact_count = exact_count
            best_total_score = total_score
            best_rule = {
                "family": "mask_repair_rule",
                "rule_type": "single_symmetry_mask_repair",
                "transform_name": candidate_name,
                "exact_count": exact_count,
                "pair_count": len(train_pairs),
                "total_score": total_score,
                "pair_results": pair_results,
            }

    return best_rule


def apply_mask_repair_rule(rule, input_grid):
    bbox = find_mask_bbox(input_grid)

    if bbox is None:
        return None

    if not is_solid_bbox(input_grid, bbox):
        return None

    return predict_from_transform(
        input_grid,
        bbox,
        rule["transform_name"],
    )


# ============================================================
# DEBUG RUNNER
# ============================================================

def main():
    task_id_or_path = input("Task id or json path: ").strip()
    task_id, task = load_task(task_id_or_path)

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    print()
    print("=" * 80)
    print(f"MASK REPAIR DEBUG: {task_id}")
    print("=" * 80)

    rule = discover_mask_repair_rule(train_pairs)

    print()
    print("BEST RULE")
    print("-" * 80)
    print(f"family        : {rule.get('family')}")
    print(f"rule_type     : {rule.get('rule_type')}")
    print(f"transform     : {rule.get('transform_name')}")
    print(f"exact_count   : {rule.get('exact_count')}/{rule.get('pair_count')}")
    print(f"total_score   : {rule.get('total_score')}")

    print()
    print("PAIR RESULTS")
    print("-" * 80)

    for pair_result in rule["pair_results"]:
        bbox = pair_result["bbox"]

        print()
        print(f"PAIR {pair_result['pair_index'] + 1}")
        print(f"bbox : {bbox}")
        print(f"score: {pair_result['score']}")
        print(f"exact: {pair_result['exact']}")

    print()
    print("=" * 80)
    print("TRAIN VISUAL CHECK")
    print("=" * 80)

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        expected_grid = pair["output"]

        predicted = apply_mask_repair_rule(rule, input_grid)

        print()
        print("-" * 80)
        print(f"TRAIN PAIR {pair_index + 1}")
        print(f"Exact: {predicted == expected_grid}")

        print_grid(input_grid, "INPUT")
        print_grid(expected_grid, "EXPECTED")

        if predicted is not None:
            print_grid(predicted, "PREDICTED")
        else:
            print("PREDICTED: None")

        if predicted is not None:
            show_three_grids(
                input_grid,
                expected_grid,
                predicted,
                title_a=f"{task_id} TRAIN {pair_index + 1} INPUT",
                title_b="EXPECTED",
                title_c="PREDICTED",
            )

    print()
    print("=" * 80)
    print("TEST PREDICTIONS")
    print("=" * 80)

    for test_index, pair in enumerate(test_pairs):
        input_grid = pair["input"]
        predicted = apply_mask_repair_rule(rule, input_grid)

        print()
        print("-" * 80)
        print(f"TEST PAIR {test_index + 1}")
        print(f"transform: {rule.get('transform_name')}")

        print_grid(input_grid, "TEST INPUT")

        if predicted is not None:
            print_grid(predicted, "TEST PREDICTED")

            show_three_grids(
                input_grid,
                predicted,
                predicted,
                title_a=f"{task_id} TEST {test_index + 1} INPUT",
                title_b="PREDICTED",
                title_c="PREDICTED",
            )
        else:
            print("TEST PREDICTED: None")


if __name__ == "__main__":
    main()