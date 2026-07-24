# debug_mask_gap_shift_learner.py
import json
import os

from core.grid_utils import print_grid
from OLD.arc_visualizer import show_three_grids


MASK_COLOR = 8
MAX_DELETE = 12
MAX_DELTA = 12


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


# ============================================================
# GRID HELPERS
# ============================================================

def grid_shape(grid):
    if grid is None:
        return "None"

    h = len(grid)
    w = len(grid[0]) if h else 0
    return f"{h}x{w}"


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


def mask_bbox_is_solid(grid, bbox):
    if bbox is None:
        return False

    for r in range(bbox["top"], bbox["bottom"] + 1):
        for c in range(bbox["left"], bbox["right"] + 1):
            if grid[r][c] != MASK_COLOR:
                return False

    return True


def edge_signature(grid, bbox):
    h = len(grid)
    w = len(grid[0]) if h else 0

    distances = {
        "top": bbox["top"],
        "left": bbox["left"],
        "bottom": h - 1 - bbox["bottom"],
        "right": w - 1 - bbox["right"],
    }

    nearest_edge = min(distances, key=distances.get)

    return {
        "grid_height": h,
        "grid_width": w,
        "nearest_edge": nearest_edge,
        "dist_top": distances["top"],
        "dist_left": distances["left"],
        "dist_bottom": distances["bottom"],
        "dist_right": distances["right"],
        "touches_top": distances["top"] == 0,
        "touches_left": distances["left"] == 0,
        "touches_bottom": distances["bottom"] == 0,
        "touches_right": distances["right"] == 0,
    }


def score_prediction(predicted, expected):
    if predicted is None:
        return 0, False

    if len(predicted) != len(expected):
        return 0, False

    if expected and len(predicted[0]) != len(expected[0]):
        return 0, False

    score = 0

    for r in range(len(expected)):
        for c in range(len(expected[0])):
            if predicted[r][c] == expected[r][c]:
                score += 1

    return score, predicted == expected


def has_mask_value(values):
    return any(value == MASK_COLOR for value in values)


# ============================================================
# GAP-LINE MODEL
# ============================================================

def remove_range(values, start, end_inclusive):
    return values[:start] + values[end_inclusive + 1:]


def apply_line_delete(values, delete_side, amount):
    if values is None:
        return None

    if amount < 0:
        return None

    if delete_side == "none":
        return values[:]

    if amount == 0:
        return values[:]

    if amount >= len(values):
        return None

    if delete_side in ("left", "top"):
        return values[amount:]

    if delete_side in ("right", "bottom"):
        return values[:-amount]

    return None


def compute_start(anchor, delta, line_length, window_size, bbox, axis):
    if line_length < window_size:
        return None

    if axis == "horizontal":
        mask_start = bbox["left"]
        mask_size = bbox["width"]
    else:
        mask_start = bbox["top"]
        mask_size = bbox["height"]

    if anchor == "line_left":
        start = delta

    elif anchor == "line_right":
        start = line_length - window_size + delta

    elif anchor == "mask_start":
        start = mask_start + delta

    elif anchor == "before_mask":
        start = mask_start - window_size + delta

    elif anchor == "after_mask":
        start = mask_start + delta

    elif anchor == "mask_start_minus_gap":
        start = mask_start - mask_size + delta

    elif anchor == "center":
        start = (line_length // 2) - (window_size // 2) + delta

    else:
        return None

    if start < 0:
        return None

    if start + window_size > line_length:
        return None

    return start


def make_horizontal_gap_prediction(input_grid, bbox, candidate):
    """
    Treat each masked row as a line.
    Remove the mask gap from that row.
    Optionally delete cells from left/right.
    Then read a learned window from the compacted line.
    """
    output = []

    for local_r in range(bbox["height"]):
        grid_r = bbox["top"] + local_r
        row = input_grid[grid_r][:]

        compact = remove_range(
            row,
            bbox["left"],
            bbox["right"],
        )

        compact = apply_line_delete(
            compact,
            candidate["delete_side"],
            candidate["delete_amount"],
        )

        if compact is None:
            return None

        start = compute_start(
            anchor=candidate["anchor"],
            delta=candidate["delta"],
            line_length=len(compact),
            window_size=bbox["width"],
            bbox=bbox,
            axis="horizontal",
        )

        if start is None:
            return None

        predicted_row = compact[start:start + bbox["width"]]

        if has_mask_value(predicted_row):
            return None

        output.append(predicted_row)

    return output


def get_column(grid, c):
    return [row[c] for row in grid]


def make_vertical_gap_prediction(input_grid, bbox, candidate):
    """
    Treat each masked column as a line.
    Remove the mask gap from that column.
    Optionally delete cells from top/bottom.
    Then read a learned window from the compacted line.
    """
    output_columns = []

    for local_c in range(bbox["width"]):
        grid_c = bbox["left"] + local_c
        column = get_column(input_grid, grid_c)

        compact = remove_range(
            column,
            bbox["top"],
            bbox["bottom"],
        )

        compact = apply_line_delete(
            compact,
            candidate["delete_side"],
            candidate["delete_amount"],
        )

        if compact is None:
            return None

        start = compute_start(
            anchor=candidate["anchor"],
            delta=candidate["delta"],
            line_length=len(compact),
            window_size=bbox["height"],
            bbox=bbox,
            axis="vertical",
        )

        if start is None:
            return None

        predicted_col = compact[start:start + bbox["height"]]

        if has_mask_value(predicted_col):
            return None

        output_columns.append(predicted_col)

    if not output_columns:
        return None

    output = []

    for r in range(bbox["height"]):
        row = []
        for c in range(bbox["width"]):
            row.append(output_columns[c][r])
        output.append(row)

    return output


def make_gap_prediction(input_grid, bbox, candidate):
    if candidate["axis"] == "horizontal":
        return make_horizontal_gap_prediction(input_grid, bbox, candidate)

    if candidate["axis"] == "vertical":
        return make_vertical_gap_prediction(input_grid, bbox, candidate)

    return None


def candidate_description(candidate):
    return (
        f"axis={candidate['axis']} "
        f"delete={candidate['delete_side']}:{candidate['delete_amount']} "
        f"anchor={candidate['anchor']} "
        f"delta={candidate['delta']}"
    )


# ============================================================
# SEARCH
# ============================================================

def generate_candidates():
    anchors = [
        "line_left",
        "line_right",
        "mask_start",
        "before_mask",
        "after_mask",
        "mask_start_minus_gap",
        "center",
    ]

    candidates = []

    for axis in ["horizontal", "vertical"]:
        if axis == "horizontal":
            delete_sides = ["none", "left", "right"]
        else:
            delete_sides = ["none", "top", "bottom"]

        for delete_side in delete_sides:
            if delete_side == "none":
                delete_amounts = [0]
            else:
                delete_amounts = list(range(1, MAX_DELETE + 1))

            for delete_amount in delete_amounts:
                for anchor in anchors:
                    for delta in range(-MAX_DELTA, MAX_DELTA + 1):
                        candidates.append({
                            "axis": axis,
                            "delete_side": delete_side,
                            "delete_amount": delete_amount,
                            "anchor": anchor,
                            "delta": delta,
                        })

    return candidates


def candidate_complexity(candidate):
    return (
        candidate["delete_amount"]
        + abs(candidate["delta"])
        + (0 if candidate["delete_side"] == "none" else 1)
        + (0 if candidate["anchor"] in ("mask_start", "after_mask") else 1)
    )


def search_gap_shifts_for_pair(input_grid, expected_grid):
    bbox = find_mask_bbox(input_grid)

    if bbox is None:
        return {
            "bbox": None,
            "edge_signature": None,
            "candidates": [],
            "best": None,
        }

    scored = []

    for candidate in generate_candidates():
        predicted = make_gap_prediction(input_grid, bbox, candidate)
        score, exact = score_prediction(predicted, expected_grid)

        scored.append({
            **candidate,
            "score": score,
            "exact": exact,
            "predicted": predicted,
            "complexity": candidate_complexity(candidate),
        })

    scored.sort(
        key=lambda item: (
            item["exact"],
            item["score"],
            -item["complexity"],
        ),
        reverse=True,
    )

    return {
        "bbox": bbox,
        "edge_signature": edge_signature(input_grid, bbox),
        "candidates": scored,
        "best": scored[0] if scored else None,
    }


# ============================================================
# LEARN SIMPLE EDGE RULE
# ============================================================

def choose_one_exact_candidate(pair_result):
    exact_candidates = [
        item for item in pair_result["candidates"]
        if item["exact"]
    ]

    if not exact_candidates:
        return None

    exact_candidates.sort(
        key=lambda item: (
            item["complexity"],
            item["axis"],
            item["delete_side"],
            item["delete_amount"],
            item["anchor"],
            item["delta"],
        )
    )

    return exact_candidates[0]


def learn_edge_gap_shift_rule(train_pair_results):
    examples = []

    for result in train_pair_results:
        chosen = choose_one_exact_candidate(result)

        if chosen is None:
            continue

        sig = result["edge_signature"]

        examples.append({
            "pair_index": result["pair_index"],
            "nearest_edge": sig["nearest_edge"],
            "axis": chosen["axis"],
            "delete_side": chosen["delete_side"],
            "delete_amount": chosen["delete_amount"],
            "anchor": chosen["anchor"],
            "delta": chosen["delta"],
            "score": chosen["score"],
        })

    if not examples:
        return None

    by_edge = {}

    for ex in examples:
        edge = ex["nearest_edge"]

        value = (
            ex["axis"],
            ex["delete_side"],
            ex["delete_amount"],
            ex["anchor"],
            ex["delta"],
        )

        if edge not in by_edge:
            by_edge[edge] = []

        by_edge[edge].append(value)

    learned_edge_transforms = {}

    for edge, values in by_edge.items():
        unique_values = sorted(set(values))

        if len(unique_values) == 1:
            learned_edge_transforms[edge] = unique_values[0]
        else:
            learned_edge_transforms[edge] = None

    consistent_edges = {
        edge: value
        for edge, value in learned_edge_transforms.items()
        if value is not None
    }

    return {
        "family": "mask_repair_rule",
        "rule_type": "learned_edge_gap_shift_mask_repair",
        "examples": examples,
        "learned_edge_transforms": learned_edge_transforms,
        "consistent_edges": consistent_edges,
        "solved_pair_count": len(examples),
        "pair_count": len(train_pair_results),
    }


def apply_learned_rule(rule, input_grid):
    bbox = find_mask_bbox(input_grid)

    if bbox is None:
        return None

    if not mask_bbox_is_solid(input_grid, bbox):
        return None

    sig = edge_signature(input_grid, bbox)
    edge = sig["nearest_edge"]

    learned_value = rule.get("consistent_edges", {}).get(edge)

    if learned_value is None:
        return None

    axis, delete_side, delete_amount, anchor, delta = learned_value

    candidate = {
        "axis": axis,
        "delete_side": delete_side,
        "delete_amount": delete_amount,
        "anchor": anchor,
        "delta": delta,
    }

    return make_gap_prediction(input_grid, bbox, candidate)


# ============================================================
# MAIN
# ============================================================

def main():
    task_id_or_path = input("Task id or json path: ").strip()
    task_id, task = load_task(task_id_or_path)

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    print()
    print("=" * 80)
    print(f"MASK GAP SHIFT LEARNER: {task_id}")
    print("=" * 80)

    train_pair_results = []

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        expected_grid = pair["output"]

        result = search_gap_shifts_for_pair(
            input_grid,
            expected_grid,
        )

        result["pair_index"] = pair_index
        result["input_grid"] = input_grid
        result["expected_grid"] = expected_grid

        train_pair_results.append(result)

        bbox = result["bbox"]
        sig = result["edge_signature"]
        best = result["best"]

        exact_candidates = [
            item for item in result["candidates"]
            if item["exact"]
        ]

        print()
        print("-" * 80)
        print(f"TRAIN PAIR {pair_index + 1}")
        print(f"input shape   : {grid_shape(input_grid)}")
        print(f"expected shape: {grid_shape(expected_grid)}")
        print(f"mask bbox     : {bbox}")
        print(f"solid mask    : {mask_bbox_is_solid(input_grid, bbox)}")
        print(f"edge sig      : {sig}")

        if best is None:
            print("BEST GAP SHIFT: None")
            continue

        print()
        print("BEST GAP SHIFT")
        print(f"transform     : {candidate_description(best)}")
        print(f"score         : {best['score']}")
        print(f"exact         : {best['exact']}")
        print(f"complexity    : {best['complexity']}")

        print()
        print(f"EXACT GAP SHIFTS FOUND: {len(exact_candidates)}")

        for idx, item in enumerate(exact_candidates[:20]):
            print(
                f"  {idx + 1}. "
                f"{candidate_description(item)} "
                f"score={item['score']} "
                f"complexity={item['complexity']}"
            )

        print_grid(expected_grid, "EXPECTED")

        if best["predicted"] is not None:
            print_grid(best["predicted"], "BEST GAP SHIFT PREDICTED")

            show_three_grids(
                input_grid,
                expected_grid,
                best["predicted"],
                title_a=f"{task_id} TRAIN {pair_index + 1} INPUT",
                title_b="EXPECTED",
                title_c="BEST GAP SHIFT PREDICTED",
            )
        else:
            print("BEST GAP SHIFT PREDICTED: None")

    print()
    print("=" * 80)
    print("LEARN EDGE GAP SHIFT RULE")
    print("=" * 80)

    learned_rule = learn_edge_gap_shift_rule(train_pair_results)

    print(learned_rule)

    if learned_rule is None:
        print("No learned rule.")
        return

    print()
    print("=" * 80)
    print("TRAIN APPLY CHECK")
    print("=" * 80)

    train_exact = 0

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        expected_grid = pair["output"]

        predicted = apply_learned_rule(
            learned_rule,
            input_grid,
        )

        score, exact = score_prediction(predicted, expected_grid)

        if exact:
            train_exact += 1

        print()
        print("-" * 80)
        print(f"TRAIN PAIR {pair_index + 1}")
        print(f"score: {score}")
        print(f"exact: {exact}")

        if predicted is not None:
            print_grid(predicted, "LEARNED RULE PREDICTED")
        else:
            print("LEARNED RULE PREDICTED: None")

    print()
    print("=" * 80)
    print(f"TRAIN EXACT: {train_exact}/{len(train_pairs)}")
    print("=" * 80)

    print()
    print("=" * 80)
    print("TEST APPLY")
    print("=" * 80)

    for test_index, pair in enumerate(test_pairs):
        input_grid = pair["input"]

        bbox = find_mask_bbox(input_grid)
        sig = edge_signature(input_grid, bbox) if bbox is not None else None

        predicted = apply_learned_rule(
            learned_rule,
            input_grid,
        )

        print()
        print("-" * 80)
        print(f"TEST PAIR {test_index + 1}")
        print(f"mask bbox : {bbox}")
        print(f"edge sig  : {sig}")
        print(f"pred shape: {grid_shape(predicted)}")

        print_grid(input_grid, "TEST INPUT")

        if predicted is not None:
            print_grid(predicted, "TEST PREDICTED")

            show_three_grids(
                input_grid,
                predicted,
                predicted,
                title_a=f"{task_id} TEST {test_index + 1} INPUT",
                title_b="TEST PREDICTED",
                title_c="TEST PREDICTED",
            )
        else:
            print("TEST PREDICTED: None")


if __name__ == "__main__":
    main()