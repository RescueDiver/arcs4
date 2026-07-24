# debug_mask_compress_search.py
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


def crop(grid, top, left, height, width):
    if grid is None:
        return None

    h = len(grid)
    w = len(grid[0]) if h else 0

    if top < 0 or left < 0:
        return None

    if top + height > h:
        return None

    if left + width > w:
        return None

    return [
        row[left:left + width]
        for row in grid[top:top + height]
    ]


def patch_has_mask(patch):
    if patch is None:
        return True

    for row in patch:
        for value in row:
            if value == MASK_COLOR:
                return True

    return False


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
# COMPRESSION TRANSFORMS
# ============================================================

def delete_left(grid, amount):
    return [
        row[amount:]
        for row in grid
    ]


def delete_right(grid, amount):
    return [
        row[:-amount]
        for row in grid
    ]


def delete_top(grid, amount):
    return grid[amount:]


def delete_bottom(grid, amount):
    return grid[:-amount]


def compress_grid(grid, delete_side, amount):
    if amount <= 0:
        return [row[:] for row in grid]

    if delete_side == "left":
        return delete_left(grid, amount)

    if delete_side == "right":
        return delete_right(grid, amount)

    if delete_side == "top":
        return delete_top(grid, amount)

    if delete_side == "bottom":
        return delete_bottom(grid, amount)

    return None


def shift_bbox_after_delete(bbox, delete_side, amount):
    if bbox is None:
        return None

    shifted = dict(bbox)

    if delete_side == "left":
        shifted["left"] -= amount
        shifted["right"] -= amount

    elif delete_side == "right":
        pass

    elif delete_side == "top":
        shifted["top"] -= amount
        shifted["bottom"] -= amount

    elif delete_side == "bottom":
        pass

    else:
        return None

    if shifted["top"] < 0 or shifted["left"] < 0:
        return None

    shifted["height"] = shifted["bottom"] - shifted["top"] + 1
    shifted["width"] = shifted["right"] - shifted["left"] + 1

    return shifted


def transform_description(candidate):
    return f"delete_{candidate['delete_side']} {candidate['amount']}"


# ============================================================
# SEARCH EXPECTED PATCH INSIDE COMPRESSED GRID
# ============================================================

def search_patch_in_grid(grid, expected_grid, allow_mask=False):
    matches = []

    grid_h = len(grid)
    grid_w = len(grid[0]) if grid_h else 0

    out_h = len(expected_grid)
    out_w = len(expected_grid[0]) if out_h else 0

    for top in range(0, grid_h - out_h + 1):
        for left in range(0, grid_w - out_w + 1):
            candidate_patch = crop(grid, top, left, out_h, out_w)

            if not allow_mask and patch_has_mask(candidate_patch):
                continue

            score, exact = score_prediction(candidate_patch, expected_grid)

            matches.append({
                "top": top,
                "left": left,
                "height": out_h,
                "width": out_w,
                "score": score,
                "exact": exact,
                "patch": candidate_patch,
            })

    matches.sort(
        key=lambda item: (
            item["exact"],
            item["score"],
            -item["top"],
            -item["left"],
        ),
        reverse=True,
    )

    return matches


def search_compressions_for_pair(input_grid, expected_grid):
    original_bbox = find_mask_bbox(input_grid)

    if original_bbox is None:
        return {
            "original_bbox": None,
            "candidates": [],
            "best": None,
        }

    candidates = []

    for delete_side in ["left", "right", "top", "bottom"]:
        for amount in range(1, MAX_DELETE + 1):
            compressed = compress_grid(
                input_grid,
                delete_side,
                amount,
            )

            if compressed is None:
                continue

            shifted_bbox = shift_bbox_after_delete(
                original_bbox,
                delete_side,
                amount,
            )

            matches = search_patch_in_grid(
                compressed,
                expected_grid,
                allow_mask=False,
            )

            if matches:
                best_match = matches[0]
            else:
                best_match = None

            exact_matches = [
                item for item in matches
                if item["exact"]
            ]

            if best_match is not None:
                if shifted_bbox is not None:
                    rel_top = best_match["top"] - shifted_bbox["top"]
                    rel_left = best_match["left"] - shifted_bbox["left"]
                else:
                    rel_top = None
                    rel_left = None

                candidates.append({
                    "delete_side": delete_side,
                    "amount": amount,
                    "compressed": compressed,
                    "shifted_bbox": shifted_bbox,
                    "best_match": best_match,
                    "exact_match_count": len(exact_matches),
                    "exact_matches": exact_matches,
                    "score": best_match["score"],
                    "exact": best_match["exact"],
                    "rel_top": rel_top,
                    "rel_left": rel_left,
                    "predicted": best_match["patch"],
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
        "original_bbox": original_bbox,
        "edge_signature": edge_signature(input_grid, original_bbox),
        "candidates": candidates,
        "best": candidates[0] if candidates else None,
    }


# ============================================================
# LEARN FROM EXACT COMPRESSION MATCHES
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
            item["delete_side"],
            abs(item["rel_top"]) if item["rel_top"] is not None else 999,
            abs(item["rel_left"]) if item["rel_left"] is not None else 999,
        )
    )

    return exact_candidates[0]


def learn_edge_compress_rule(train_pair_results):
    examples = []

    for result in train_pair_results:
        chosen = choose_one_exact_candidate(result)

        if chosen is None:
            continue

        sig = result["edge_signature"]

        examples.append({
            "pair_index": result["pair_index"],
            "nearest_edge": sig["nearest_edge"],
            "delete_side": chosen["delete_side"],
            "amount": chosen["amount"],
            "rel_top": chosen["rel_top"],
            "rel_left": chosen["rel_left"],
            "score": chosen["score"],
        })

    if not examples:
        return None

    by_edge = {}

    for ex in examples:
        edge = ex["nearest_edge"]

        value = (
            ex["delete_side"],
            ex["amount"],
            ex["rel_top"],
            ex["rel_left"],
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
        "rule_type": "learned_edge_compress_search_mask_repair",
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

    sig = edge_signature(input_grid, bbox)
    edge = sig["nearest_edge"]

    learned_value = rule.get("consistent_edges", {}).get(edge)

    if learned_value is None:
        return None

    delete_side, amount, rel_top, rel_left = learned_value

    compressed = compress_grid(
        input_grid,
        delete_side,
        amount,
    )

    shifted_bbox = shift_bbox_after_delete(
        bbox,
        delete_side,
        amount,
    )

    if compressed is None or shifted_bbox is None:
        return None

    top = shifted_bbox["top"] + rel_top
    left = shifted_bbox["left"] + rel_left

    return crop(
        compressed,
        top,
        left,
        bbox["height"],
        bbox["width"],
    )


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
    print(f"MASK COMPRESS SEARCH: {task_id}")
    print("=" * 80)

    train_pair_results = []

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        expected_grid = pair["output"]

        result = search_compressions_for_pair(
            input_grid,
            expected_grid,
        )

        result["pair_index"] = pair_index
        result["input_grid"] = input_grid
        result["expected_grid"] = expected_grid

        train_pair_results.append(result)

        bbox = result["original_bbox"]
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
            print("BEST COMPRESS MATCH: None")
            continue

        print()
        print("BEST COMPRESS MATCH")
        print(f"transform     : {transform_description(best)}")
        print(f"compressed    : {grid_shape(best['compressed'])}")
        print(f"shifted bbox  : {best['shifted_bbox']}")
        print(f"found top/left: {best['best_match']['top']}, {best['best_match']['left']}")
        print(f"rel to bbox   : {best['rel_top']}, {best['rel_left']}")
        print(f"score         : {best['score']}")
        print(f"exact         : {best['exact']}")

        print()
        print(f"EXACT COMPRESS MATCHES FOUND: {len(exact_candidates)}")

        for idx, item in enumerate(exact_candidates[:20]):
            print(
                f"  {idx + 1}. "
                f"{transform_description(item)} "
                f"found=({item['best_match']['top']}, {item['best_match']['left']}) "
                f"rel=({item['rel_top']}, {item['rel_left']}) "
                f"score={item['score']}"
            )

        print_grid(expected_grid, "EXPECTED")

        if best["predicted"] is not None:
            print_grid(best["predicted"], "BEST COMPRESS PREDICTED")

            show_three_grids(
                input_grid,
                expected_grid,
                best["predicted"],
                title_a=f"{task_id} TRAIN {pair_index + 1} INPUT",
                title_b="EXPECTED",
                title_c="BEST COMPRESS PREDICTED",
            )
        else:
            print("BEST COMPRESS PREDICTED: None")

    print()
    print("=" * 80)
    print("LEARN EDGE COMPRESS RULE")
    print("=" * 80)

    learned_rule = learn_edge_compress_rule(train_pair_results)

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