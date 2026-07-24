# debug_mask_offset_learner.py
import json
import os

from core.grid_utils import print_grid
from OLD.arc_visualizer import show_three_grids


MASK_COLOR = 8
MAX_OFFSET = 12


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


def mask_bbox_is_solid(grid, bbox, mask_color=MASK_COLOR):
    if bbox is None:
        return False

    for r in range(bbox["top"], bbox["bottom"] + 1):
        for c in range(bbox["left"], bbox["right"] + 1):
            if grid[r][c] != mask_color:
                return False

    return True


def get_grid_value(grid, r, c):
    h = len(grid)
    w = len(grid[0]) if h else 0

    if r < 0 or c < 0 or r >= h or c >= w:
        return None

    return grid[r][c]


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


# ============================================================
# OFFSET PATCH GENERATION
# ============================================================

def predict_mask_patch_by_offset(input_grid, bbox, row_delta, col_delta):
    if bbox is None:
        return None

    output = []

    for local_r in range(bbox["height"]):
        row = []

        for local_c in range(bbox["width"]):
            mask_r = bbox["top"] + local_r
            mask_c = bbox["left"] + local_c

            source_r = mask_r + row_delta
            source_c = mask_c + col_delta

            value = get_grid_value(input_grid, source_r, source_c)

            if value is None:
                return None

            if value == MASK_COLOR:
                return None

            row.append(value)

        output.append(row)

    return output


def search_offsets_for_pair(input_grid, expected_grid, max_offset=MAX_OFFSET):
    bbox = find_mask_bbox(input_grid)

    if bbox is None:
        return {
            "bbox": None,
            "offsets": [],
            "best": None,
        }

    offsets = []

    for row_delta in range(-max_offset, max_offset + 1):
        for col_delta in range(-max_offset, max_offset + 1):
            if row_delta == 0 and col_delta == 0:
                continue

            predicted = predict_mask_patch_by_offset(
                input_grid,
                bbox,
                row_delta,
                col_delta,
            )

            score, exact = score_prediction(predicted, expected_grid)

            offsets.append({
                "row_delta": row_delta,
                "col_delta": col_delta,
                "score": score,
                "exact": exact,
                "predicted": predicted,
            })

    offsets.sort(
        key=lambda item: (
            item["exact"],
            item["score"],
            -abs(item["row_delta"]) - abs(item["col_delta"]),
        ),
        reverse=True,
    )

    best = offsets[0] if offsets else None

    return {
        "bbox": bbox,
        "offsets": offsets,
        "best": best,
    }


# ============================================================
# LEARNING OFFSET RELATION
# ============================================================

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


def choose_one_exact_offset(pair_result):
    exact_offsets = [
        item for item in pair_result["offsets"]
        if item["exact"]
    ]

    if not exact_offsets:
        return None

    exact_offsets.sort(
        key=lambda item: (
            abs(item["row_delta"]) + abs(item["col_delta"]),
            abs(item["row_delta"]),
            abs(item["col_delta"]),
        )
    )

    return exact_offsets[0]


def learn_edge_offset_rule(train_pair_results):
    """
    This does not hardcode offset 2.

    It looks at train pairs that were solved exactly by offsets,
    then asks whether the chosen offset is predictable from the mask's nearest edge.

    Learned examples:
        nearest right edge -> col_delta +2
        nearest left edge  -> col_delta -2
        nearest top edge   -> row_delta -2
        nearest bottom edge-> row_delta +2

    But those values only come from train data.
    """

    examples = []

    for result in train_pair_results:
        exact_offset = choose_one_exact_offset(result)

        if exact_offset is None:
            continue

        sig = result["edge_signature"]

        examples.append({
            "pair_index": result["pair_index"],
            "nearest_edge": sig["nearest_edge"],
            "row_delta": exact_offset["row_delta"],
            "col_delta": exact_offset["col_delta"],
            "score": exact_offset["score"],
        })

    if not examples:
        return None

    by_edge = {}

    for ex in examples:
        edge = ex["nearest_edge"]
        offset = (ex["row_delta"], ex["col_delta"])

        if edge not in by_edge:
            by_edge[edge] = []

        by_edge[edge].append(offset)

    learned_edge_offsets = {}

    for edge, offsets in by_edge.items():
        unique_offsets = sorted(set(offsets))

        if len(unique_offsets) == 1:
            learned_edge_offsets[edge] = unique_offsets[0]
        else:
            learned_edge_offsets[edge] = None

    consistent_edges = {
        edge: offset
        for edge, offset in learned_edge_offsets.items()
        if offset is not None
    }

    return {
        "family": "mask_repair_rule",
        "rule_type": "learned_edge_offset_mask_repair",
        "examples": examples,
        "learned_edge_offsets": learned_edge_offsets,
        "consistent_edges": consistent_edges,
        "solved_pair_count": len(examples),
        "pair_count": len(train_pair_results),
    }


def apply_learned_edge_offset_rule(rule, input_grid):
    bbox = find_mask_bbox(input_grid)

    if bbox is None:
        return None

    if not mask_bbox_is_solid(input_grid, bbox):
        return None

    sig = edge_signature(input_grid, bbox)
    edge = sig["nearest_edge"]

    offset = rule.get("consistent_edges", {}).get(edge)

    if offset is None:
        return None

    row_delta, col_delta = offset

    return predict_mask_patch_by_offset(
        input_grid,
        bbox,
        row_delta,
        col_delta,
    )


# ============================================================
# MAIN DEBUG
# ============================================================

def main():
    task_id_or_path = input("Task id or json path: ").strip()
    task_id, task = load_task(task_id_or_path)

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    print()
    print("=" * 80)
    print(f"MASK OFFSET LEARNER: {task_id}")
    print("=" * 80)

    train_pair_results = []

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        expected_grid = pair["output"]

        result = search_offsets_for_pair(
            input_grid,
            expected_grid,
            max_offset=MAX_OFFSET,
        )

        bbox = result["bbox"]
        sig = edge_signature(input_grid, bbox) if bbox is not None else None
        best = result["best"]
        exact_offsets = [
            item for item in result["offsets"]
            if item["exact"]
        ]

        result["pair_index"] = pair_index
        result["input_grid"] = input_grid
        result["expected_grid"] = expected_grid
        result["edge_signature"] = sig
        train_pair_results.append(result)

        print()
        print("-" * 80)
        print(f"TRAIN PAIR {pair_index + 1}")
        print(f"input shape   : {grid_shape(input_grid)}")
        print(f"expected shape: {grid_shape(expected_grid)}")
        print(f"mask bbox     : {bbox}")
        print(f"solid mask    : {mask_bbox_is_solid(input_grid, bbox)}")
        print(f"edge sig      : {sig}")

        if best is None:
            print("best offset   : None")
            continue

        print()
        print("BEST OFFSET")
        print(
            f"offset        : row {best['row_delta']}, "
            f"col {best['col_delta']}"
        )
        print(f"score         : {best['score']}")
        print(f"exact         : {best['exact']}")

        print()
        print(f"EXACT OFFSETS FOUND: {len(exact_offsets)}")

        for idx, item in enumerate(exact_offsets[:20]):
            print(
                f"  {idx + 1}. row={item['row_delta']} "
                f"col={item['col_delta']} "
                f"score={item['score']}"
            )

        print_grid(expected_grid, "EXPECTED")

        if best["predicted"] is not None:
            print_grid(best["predicted"], "BEST OFFSET PREDICTED")

            show_three_grids(
                input_grid,
                expected_grid,
                best["predicted"],
                title_a=f"{task_id} TRAIN {pair_index + 1} INPUT",
                title_b="EXPECTED",
                title_c="BEST OFFSET PREDICTED",
            )
        else:
            print("BEST OFFSET PREDICTED: None")

    print()
    print("=" * 80)
    print("LEARN EDGE OFFSET RULE")
    print("=" * 80)

    learned_rule = learn_edge_offset_rule(train_pair_results)

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

        predicted = apply_learned_edge_offset_rule(
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

        predicted = apply_learned_edge_offset_rule(
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