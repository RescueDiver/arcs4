# debug_mask_canvas_shift_learner.py
import json
import os

from core.grid_utils import print_grid
from OLD.arc_visualizer import show_three_grids


MASK_COLOR = 8
MAX_SHIFT = 12


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
# BASIC GRID HELPERS
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


def crop_bbox(grid, bbox):
    if grid is None or bbox is None:
        return None

    h = len(grid)
    w = len(grid[0]) if h else 0

    if bbox["bottom"] >= h:
        return None

    if bbox["right"] >= w:
        return None

    return [
        row[bbox["left"]:bbox["right"] + 1]
        for row in grid[bbox["top"]:bbox["bottom"] + 1]
    ]


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


# ============================================================
# CANVAS SHIFT TRANSFORMS
# ============================================================

def make_blank_row(width, value=MASK_COLOR):
    return [value for _ in range(width)]


def shift_canvas_delete_left(grid, n, pad_mode):
    shifted = []

    for row in grid:
        kept = row[n:]

        if pad_mode == "blank":
            pad = [MASK_COLOR for _ in range(n)]
        elif pad_mode == "wrap":
            pad = row[:n]
        elif pad_mode == "edge":
            pad_value = row[-1] if row else MASK_COLOR
            pad = [pad_value for _ in range(n)]
        else:
            return None

        shifted.append(kept + pad)

    return shifted


def shift_canvas_delete_right(grid, n, pad_mode):
    shifted = []

    for row in grid:
        kept = row[:-n] if n else row[:]

        if pad_mode == "blank":
            pad = [MASK_COLOR for _ in range(n)]
        elif pad_mode == "wrap":
            pad = row[-n:] if n else []
        elif pad_mode == "edge":
            pad_value = row[0] if row else MASK_COLOR
            pad = [pad_value for _ in range(n)]
        else:
            return None

        shifted.append(pad + kept)

    return shifted


def shift_canvas_delete_top(grid, n, pad_mode):
    if not grid:
        return []

    width = len(grid[0])
    kept = grid[n:]

    if pad_mode == "blank":
        pad = [make_blank_row(width) for _ in range(n)]
    elif pad_mode == "wrap":
        pad = grid[:n]
    elif pad_mode == "edge":
        pad = [grid[-1][:] for _ in range(n)]
    else:
        return None

    return kept + pad


def shift_canvas_delete_bottom(grid, n, pad_mode):
    if not grid:
        return []

    width = len(grid[0])
    kept = grid[:-n] if n else [row[:] for row in grid]

    if pad_mode == "blank":
        pad = [make_blank_row(width) for _ in range(n)]
    elif pad_mode == "wrap":
        pad = grid[-n:] if n else []
    elif pad_mode == "edge":
        pad = [grid[0][:] for _ in range(n)]
    else:
        return None

    return pad + kept


def transform_canvas(grid, transform_name, amount, pad_mode):
    if amount <= 0:
        return [row[:] for row in grid]

    if transform_name == "delete_left":
        return shift_canvas_delete_left(grid, amount, pad_mode)

    if transform_name == "delete_right":
        return shift_canvas_delete_right(grid, amount, pad_mode)

    if transform_name == "delete_top":
        return shift_canvas_delete_top(grid, amount, pad_mode)

    if transform_name == "delete_bottom":
        return shift_canvas_delete_bottom(grid, amount, pad_mode)

    return None


def transform_description(candidate):
    return (
        f"{candidate['transform_name']} "
        f"{candidate['amount']} "
        f"pad={candidate['pad_mode']}"
    )


# ============================================================
# SEARCH
# ============================================================

def search_canvas_shifts_for_pair(input_grid, expected_grid):
    bbox = find_mask_bbox(input_grid)

    if bbox is None:
        return {
            "bbox": None,
            "edge_signature": None,
            "candidates": [],
            "best": None,
        }

    candidates = []

    transform_names = [
        "delete_left",
        "delete_right",
        "delete_top",
        "delete_bottom",
    ]

    pad_modes = [
        "blank",
        "wrap",
        "edge",
    ]

    for transform_name in transform_names:
        for amount in range(1, MAX_SHIFT + 1):
            for pad_mode in pad_modes:
                transformed = transform_canvas(
                    input_grid,
                    transform_name,
                    amount,
                    pad_mode,
                )

                predicted = crop_bbox(transformed, bbox)
                score, exact = score_prediction(predicted, expected_grid)

                candidates.append({
                    "transform_name": transform_name,
                    "amount": amount,
                    "pad_mode": pad_mode,
                    "score": score,
                    "exact": exact,
                    "predicted": predicted,
                    "transformed": transformed,
                })

    candidates.sort(
        key=lambda item: (
            item["exact"],
            item["score"],
            -item["amount"],
        ),
        reverse=True,
    )

    return {
        "bbox": bbox,
        "edge_signature": edge_signature(input_grid, bbox),
        "candidates": candidates,
        "best": candidates[0] if candidates else None,
    }


# ============================================================
# LEARNING
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
            item["amount"],
            item["transform_name"],
            item["pad_mode"],
        )
    )

    return exact_candidates[0]


def learn_edge_canvas_shift_rule(train_pair_results):
    examples = []

    for result in train_pair_results:
        chosen = choose_one_exact_candidate(result)

        if chosen is None:
            continue

        sig = result["edge_signature"]

        examples.append({
            "pair_index": result["pair_index"],
            "nearest_edge": sig["nearest_edge"],
            "transform_name": chosen["transform_name"],
            "amount": chosen["amount"],
            "pad_mode": chosen["pad_mode"],
            "score": chosen["score"],
        })

    if not examples:
        return None

    by_edge = {}

    for ex in examples:
        edge = ex["nearest_edge"]
        learned_value = (
            ex["transform_name"],
            ex["amount"],
            ex["pad_mode"],
        )

        if edge not in by_edge:
            by_edge[edge] = []

        by_edge[edge].append(learned_value)

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
        "rule_type": "learned_edge_canvas_shift_mask_repair",
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

    transform_info = rule.get("consistent_edges", {}).get(edge)

    if transform_info is None:
        return None

    transform_name, amount, pad_mode = transform_info

    transformed = transform_canvas(
        input_grid,
        transform_name,
        amount,
        pad_mode,
    )

    return crop_bbox(transformed, bbox)


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
    print(f"MASK CANVAS SHIFT LEARNER: {task_id}")
    print("=" * 80)

    train_pair_results = []

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        expected_grid = pair["output"]

        result = search_canvas_shifts_for_pair(
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
            print("BEST CANVAS SHIFT: None")
            continue

        print()
        print("BEST CANVAS SHIFT")
        print(f"transform     : {transform_description(best)}")
        print(f"score         : {best['score']}")
        print(f"exact         : {best['exact']}")

        print()
        print(f"EXACT CANVAS SHIFTS FOUND: {len(exact_candidates)}")

        for idx, item in enumerate(exact_candidates[:20]):
            print(
                f"  {idx + 1}. "
                f"{transform_description(item)} "
                f"score={item['score']}"
            )

        print_grid(expected_grid, "EXPECTED")

        if best["predicted"] is not None:
            print_grid(best["predicted"], "BEST CANVAS SHIFT PREDICTED")

            show_three_grids(
                input_grid,
                expected_grid,
                best["predicted"],
                title_a=f"{task_id} TRAIN {pair_index + 1} INPUT",
                title_b="EXPECTED",
                title_c="BEST CANVAS SHIFT PREDICTED",
            )
        else:
            print("BEST CANVAS SHIFT PREDICTED: None")

    print()
    print("=" * 80)
    print("LEARN EDGE CANVAS SHIFT RULE")
    print("=" * 80)

    learned_rule = learn_edge_canvas_shift_rule(train_pair_results)

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