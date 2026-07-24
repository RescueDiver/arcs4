# debug_mask_predelete_shared_learner.py
import json
import os
import sys

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


def edge_group_from_signature(sig):
    edge = sig["nearest_edge"]

    if edge in ("left", "right"):
        return "left_right_edge_group"

    if edge in ("top", "bottom"):
        return "top_bottom_edge_group"

    return None


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


# ============================================================
# CANDIDATES
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


def candidate_key(candidate):
    return (
        candidate["delete_side"],
        candidate["delete_amount"],
        candidate["transform_name"],
    )


def key_to_candidate(key):
    delete_side, delete_amount, transform_name = key

    return {
        "delete_side": delete_side,
        "delete_amount": delete_amount,
        "transform_name": transform_name,
    }


def candidate_description(candidate):
    return (
        f"delete_{candidate['delete_side']} "
        f"{candidate['delete_amount']} "
        f"then {candidate['transform_name']}"
    )


def key_description(key):
    return candidate_description(key_to_candidate(key))


def candidate_complexity(candidate):
    return (
        candidate["delete_amount"]
        + (0 if candidate["delete_side"] == "none" else 1)
        + (0 if candidate["transform_name"] == "identity" else 1)
    )


def key_complexity(key):
    return candidate_complexity(key_to_candidate(key))


# ============================================================
# SEARCH EACH TRAIN PAIR
# ============================================================

def search_pair(input_grid, expected_grid):
    bbox = find_mask_bbox(input_grid)

    if bbox is None:
        return {
            "bbox": None,
            "edge_signature": None,
            "edge_group": None,
            "candidates": [],
            "best": None,
        }

    sig = edge_signature(input_grid, bbox)
    edge_group = edge_group_from_signature(sig)

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
        "edge_signature": sig,
        "edge_group": edge_group,
        "candidates": scored,
        "best": scored[0] if scored else None,
    }


def exact_candidate_keys(pair_result):
    keys = set()

    for item in pair_result["candidates"]:
        if item["exact"]:
            keys.add(candidate_key(item))

    return keys


# ============================================================
# LEARN SHARED EXACT RULE
# ============================================================

def learn_shared_edge_group_rule(train_pair_results):
    groups = {}

    for result in train_pair_results:
        group = result["edge_group"]

        if group is None:
            continue

        if group not in groups:
            groups[group] = []

        groups[group].append(result)

    learned_group_keys = {}
    group_debug = {}

    for group, results in groups.items():
        exact_sets = [
            exact_candidate_keys(result)
            for result in results
        ]

        if not exact_sets:
            learned_group_keys[group] = None
            continue

        shared = set(exact_sets[0])

        for exact_set in exact_sets[1:]:
            shared = shared.intersection(exact_set)

        sorted_shared = sorted(
            shared,
            key=lambda key: (
                key_complexity(key),
                key[0],
                key[1],
                key[2],
            )
        )

        chosen = sorted_shared[0] if sorted_shared else None

        learned_group_keys[group] = chosen

        group_debug[group] = {
            "example_count": len(results),
            "shared_exact_count": len(sorted_shared),
            "shared_exact": sorted_shared,
            "chosen": chosen,
        }

    all_pair_indices = [
        result["pair_index"]
        for result in train_pair_results
    ]

    solved_pair_indices = []

    for result in train_pair_results:
        group = result["edge_group"]
        learned_key = learned_group_keys.get(group)

        if learned_key is None:
            continue

        if learned_key in exact_candidate_keys(result):
            solved_pair_indices.append(result["pair_index"])

    return {
        "family": "mask_repair_rule",
        "rule_type": "learned_shared_predelete_transform_by_edge_group",
        "learned_group_keys": learned_group_keys,
        "group_debug": group_debug,
        "all_pair_indices": all_pair_indices,
        "solved_pair_indices": solved_pair_indices,
        "train_pair_count": len(train_pair_results),
        "train_solved_by_shared_rule_count": len(solved_pair_indices),
    }


def opposite_delete_side(delete_side):
    if delete_side == "left":
        return "right"

    if delete_side == "right":
        return "left"

    if delete_side == "top":
        return "bottom"

    if delete_side == "bottom":
        return "top"

    return delete_side


def prediction_is_valid(predicted):
    if predicted is None:
        return False

    for row in predicted:
        for value in row:
            if value == MASK_COLOR:
                return False

    return True


def apply_candidate_to_input(candidate, input_grid, bbox):
    return predict_from_predelete_transform(
        input_grid,
        bbox,
        candidate,
    )


def apply_learned_rule(rule, input_grid):
    bbox = find_mask_bbox(input_grid)

    if bbox is None:
        return None

    if not mask_bbox_is_solid(input_grid, bbox):
        return None

    sig = edge_signature(input_grid, bbox)
    group = edge_group_from_signature(sig)

    learned_key = rule["learned_group_keys"].get(group)

    if learned_key is None:
        return None

    learned_candidate = key_to_candidate(learned_key)

    # First try the exact learned candidate.
    predicted = apply_candidate_to_input(
        learned_candidate,
        input_grid,
        bbox,
    )

    if prediction_is_valid(predicted):
        return predicted

    # If the learned delete side breaks the bbox on a new edge,
    # try the symmetric delete side with the SAME learned amount
    # and SAME learned transform.
    flipped_candidate = dict(learned_candidate)
    flipped_candidate["delete_side"] = opposite_delete_side(
        learned_candidate["delete_side"]
    )

    predicted = apply_candidate_to_input(
        flipped_candidate,
        input_grid,
        bbox,
    )

    if prediction_is_valid(predicted):
        return predicted

    return None


# ============================================================
# PRINT HELPERS
# ============================================================

def print_pair_search_result(task_id, pair_index, input_grid, expected_grid, result):
    bbox = result["bbox"]
    sig = result["edge_signature"]
    edge_group = result["edge_group"]
    best = result["best"]

    print()
    print("-" * 80)
    print(f"TRAIN PAIR {pair_index + 1}")
    print(f"input shape   : {grid_shape(input_grid)}")
    print(f"expected shape: {grid_shape(expected_grid)}")
    print(f"mask bbox     : {bbox}")
    print(f"solid mask    : {mask_bbox_is_solid(input_grid, bbox)}")
    print(f"edge sig      : {sig}")
    print(f"edge group    : {edge_group}")

    if bbox is None:
        print()
        print("SKIP: no mask bbox found. This is not a mask_repair_rule task.")
        return

    if best is None:
        print()
        print("SKIP: no predelete/transform candidates found.")
        return

    exact_items = [
        item for item in result["candidates"]
        if item["exact"]
    ]

    print()
    print("BEST PREDELETE + TRANSFORM")
    print(f"transform     : {candidate_description(best)}")
    print(f"score         : {best['score']}")
    print(f"exact         : {best['exact']}")
    print(f"complexity    : {best['complexity']}")

    print()
    print(f"EXACT PREDELETE + TRANSFORM FOUND: {len(exact_items)}")

    for idx, item in enumerate(exact_items[:30]):
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


def print_learned_rule(rule):
    print()
    print("=" * 80)
    print("LEARNED SHARED EDGE-GROUP RULE")
    print("=" * 80)

    print(f"rule family : {rule['family']}")
    print(f"rule type   : {rule['rule_type']}")
    print(
        "train solved by shared rule: "
        f"{rule['train_solved_by_shared_rule_count']}/"
        f"{rule['train_pair_count']}"
    )

    print()
    print("GROUPS")

    for group, debug in rule["group_debug"].items():
        chosen = debug["chosen"]

        print()
        print(f"{group}")
        print(f"  examples          : {debug['example_count']}")
        print(f"  shared exact count: {debug['shared_exact_count']}")

        if chosen is None:
            print("  chosen            : None")
        else:
            print(f"  chosen            : {key_description(chosen)}")

        for idx, key in enumerate(debug["shared_exact"][:20]):
            print(f"    {idx + 1}. {key_description(key)}")


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) >= 2:
        task_id_or_path = sys.argv[1]
    else:
        task_id_or_path = input("Task id or json path: ").strip()

    task_id, task = load_task(task_id_or_path)

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    print()
    print("=" * 80)
    print(f"MASK PREDELETE SHARED LEARNER: {task_id}")
    print("=" * 80)

    train_pair_results = []

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        expected_grid = pair["output"]

        result = search_pair(input_grid, expected_grid)
        result["pair_index"] = pair_index
        result["input_grid"] = input_grid
        result["expected_grid"] = expected_grid

        train_pair_results.append(result)

        print_pair_search_result(
            task_id,
            pair_index,
            input_grid,
            expected_grid,
            result,
        )

    learned_rule = learn_shared_edge_group_rule(train_pair_results)

    print_learned_rule(learned_rule)

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

            show_three_grids(
                input_grid,
                expected_grid,
                predicted,
                title_a=f"{task_id} TRAIN {pair_index + 1} INPUT",
                title_b="EXPECTED",
                title_c="LEARNED RULE PREDICTED",
            )
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
        edge_group = edge_group_from_signature(sig) if sig is not None else None

        predicted = apply_learned_rule(
            learned_rule,
            input_grid,
        )

        print()
        print("-" * 80)
        print(f"TEST PAIR {test_index + 1}")
        print(f"mask bbox : {bbox}")
        print(f"edge sig  : {sig}")
        print(f"edge group: {edge_group}")
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