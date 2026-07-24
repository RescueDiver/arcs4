# debug_mask_predelete_transform_learner.py
import json
import os

from core.grid_utils import print_grid
from OLD.arc_visualizer import show_three_grids


MASK_COLOR = 8
MAX_DELETE = 12


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


def find_mask_bbox(grid):
    cells = []

    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == MASK_COLOR:
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


def get_value(grid, r, c):
    h = len(grid)
    w = len(grid[0]) if h else 0

    if r < 0 or c < 0 or r >= h or c >= w:
        return None

    value = grid[r][c]

    if value == MASK_COLOR:
        return None

    return value


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
# PREDELETE
# ============================================================

def delete_grid_side(grid, side, amount):
    if side == "none" or amount == 0:
        return [row[:] for row in grid]

    if side == "left":
        return [row[amount:] for row in grid]

    if side == "right":
        return [row[:-amount] for row in grid]

    if side == "top":
        return grid[amount:]

    if side == "bottom":
        return grid[:-amount]

    return None


def shift_bbox_after_delete(bbox, side, amount):
    shifted = dict(bbox)

    if side == "left":
        shifted["left"] -= amount
        shifted["right"] -= amount

    elif side == "top":
        shifted["top"] -= amount
        shifted["bottom"] -= amount

    elif side in ("right", "bottom", "none"):
        pass

    else:
        return None

    if shifted["top"] < 0 or shifted["left"] < 0:
        return None

    shifted["height"] = shifted["bottom"] - shifted["top"] + 1
    shifted["width"] = shifted["right"] - shifted["left"] + 1

    return shifted


def bbox_fits_grid(grid, bbox):
    if grid is None or bbox is None:
        return False

    h = len(grid)
    w = len(grid[0]) if h else 0

    return (
        bbox["top"] >= 0
        and bbox["left"] >= 0
        and bbox["bottom"] < h
        and bbox["right"] < w
    )


# ============================================================
# TRANSFORMS
# ============================================================

def source_coord(transform_name, grid_h, grid_w, r, c):
    if transform_name == "identity":
        return r, c

    if transform_name == "vertical_mirror":
        return r, grid_w - 1 - c

    if transform_name == "horizontal_mirror":
        return grid_h - 1 - r, c

    if transform_name == "both_mirror":
        return grid_h - 1 - r, grid_w - 1 - c

    if transform_name == "main_diagonal":
        return c, r

    if transform_name == "anti_diagonal":
        return grid_w - 1 - c, grid_h - 1 - r

    if transform_name == "rotate_90_position":
        return c, grid_h - 1 - r

    if transform_name == "rotate_270_position":
        return grid_w - 1 - c, r

    return None, None


def predict_from_predelete_transform(input_grid, original_bbox, candidate):
    side = candidate["delete_side"]
    amount = candidate["delete_amount"]
    transform_name = candidate["transform_name"]

    transformed_grid = delete_grid_side(input_grid, side, amount)

    if transformed_grid is None:
        return None

    shifted_bbox = shift_bbox_after_delete(
        original_bbox,
        side,
        amount,
    )

    if not bbox_fits_grid(transformed_grid, shifted_bbox):
        return None

    grid_h = len(transformed_grid)
    grid_w = len(transformed_grid[0]) if grid_h else 0

    output = []

    for local_r in range(shifted_bbox["height"]):
        row = []

        for local_c in range(shifted_bbox["width"]):
            r = shifted_bbox["top"] + local_r
            c = shifted_bbox["left"] + local_c

            source_r, source_c = source_coord(
                transform_name,
                grid_h,
                grid_w,
                r,
                c,
            )

            value = get_value(
                transformed_grid,
                source_r,
                source_c,
            )

            if value is None:
                return None

            row.append(value)

        output.append(row)

    return output


def candidate_description(candidate):
    return (
        f"delete_{candidate['delete_side']} "
        f"{candidate['delete_amount']} "
        f"then {candidate['transform_name']}"
    )


# ============================================================
# SEARCH
# ============================================================

def generate_candidates():
    transforms = [
        "identity",
        "vertical_mirror",
        "horizontal_mirror",
        "both_mirror",
        "main_diagonal",
        "anti_diagonal",
        "rotate_90_position",
        "rotate_270_position",
    ]

    candidates = []

    for transform_name in transforms:
        candidates.append({
            "delete_side": "none",
            "delete_amount": 0,
            "transform_name": transform_name,
        })

    for delete_side in ["left", "right", "top", "bottom"]:
        for amount in range(1, MAX_DELETE + 1):
            for transform_name in transforms:
                candidates.append({
                    "delete_side": delete_side,
                    "delete_amount": amount,
                    "transform_name": transform_name,
                })

    return candidates


def candidate_complexity(candidate):
    return (
        candidate["delete_amount"]
        + (0 if candidate["delete_side"] == "none" else 1)
        + (0 if candidate["transform_name"] == "identity" else 1)
    )


def search_pair(input_grid, expected_grid):
    bbox = find_mask_bbox(input_grid)

    scored = []

    for candidate in generate_candidates():
        predicted = predict_from_predelete_transform(
            input_grid,
            bbox,
            candidate,
        )

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
# MAIN
# ============================================================

def main():
    task_id_or_path = input("Task id or json path: ").strip()
    task_id, task = load_task(task_id_or_path)

    train_pairs = task.get("train", [])

    print()
    print("=" * 80)
    print(f"MASK PREDELETE + TRANSFORM LEARNER: {task_id}")
    print("=" * 80)

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        expected_grid = pair["output"]

        result = search_pair(input_grid, expected_grid)

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

        print()
        print("BEST PREDELETE + TRANSFORM")
        print(f"transform     : {candidate_description(best)}")
        print(f"score         : {best['score']}")
        print(f"exact         : {best['exact']}")
        print(f"complexity    : {best['complexity']}")

        print()
        print(f"EXACT PREDELETE + TRANSFORM FOUND: {len(exact_candidates)}")

        for idx, item in enumerate(exact_candidates[:30]):
            print(
                f"  {idx + 1}. "
                f"{candidate_description(item)} "
                f"score={item['score']} "
                f"complexity={item['complexity']}"
            )

        print_grid(expected_grid, "EXPECTED")

        if best["predicted"] is not None:
            print_grid(best["predicted"], "BEST PREDICTED")

            show_three_grids(
                input_grid,
                expected_grid,
                best["predicted"],
                title_a=f"{task_id} TRAIN {pair_index + 1} INPUT",
                title_b="EXPECTED",
                title_c="BEST PREDICTED",
            )
        else:
            print("BEST PREDICTED: None")


if __name__ == "__main__":
    main()