# reasoning/mask_repair_rule.py

MASK_COLOR = 8
MAX_DELETE = 12


# ============================================================
# BASIC HELPERS
# ============================================================

def grid_shape(grid):
    if grid is None:
        return None

    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


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


# ============================================================
# EDGE / GROUP DETECTION
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
# CANDIDATE SEARCH
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


def candidate_complexity(candidate):
    return (
        candidate["delete_amount"]
        + (0 if candidate["delete_side"] == "none" else 1)
        + (0 if candidate["transform_name"] == "identity" else 1)
    )


def key_complexity(key):
    return candidate_complexity(key_to_candidate(key))


def exact_candidate_keys_for_pair(input_grid, expected_grid):
    bbox = find_mask_bbox(input_grid)

    if bbox is None:
        return set(), None, None

    if not mask_bbox_is_solid(input_grid, bbox):
        return set(), None, None

    sig = edge_signature(input_grid, bbox)
    group = edge_group_from_signature(sig)

    exact_keys = set()

    for candidate in generate_candidates():
        predicted = predict_from_predelete_transform(
            input_grid,
            bbox,
            candidate,
        )

        _, exact = score_prediction(predicted, expected_grid)

        if exact:
            exact_keys.add(candidate_key(candidate))

    return exact_keys, sig, group


# ============================================================
# LEARN RULE
# ============================================================

def learn_mask_repair_rule(train_pairs):
    group_exact_sets = {}
    group_examples = {}
    pair_results = []

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        expected_grid = pair["output"]

        exact_keys, sig, group = exact_candidate_keys_for_pair(
            input_grid,
            expected_grid,
        )

        if group is None:
            return None

        if not exact_keys:
            return None

        if group not in group_exact_sets:
            group_exact_sets[group] = []

        if group not in group_examples:
            group_examples[group] = []

        group_exact_sets[group].append(exact_keys)
        group_examples[group].append(pair_index)

        pair_results.append({
            "pair_index": pair_index,
            "edge_signature": sig,
            "edge_group": group,
            "exact_count": len(exact_keys),
        })

    learned_group_keys = {}
    group_debug = {}

    for group, exact_sets in group_exact_sets.items():
        shared = set(exact_sets[0])

        for exact_set in exact_sets[1:]:
            shared = shared.intersection(exact_set)

        if not shared:
            return None

        sorted_shared = sorted(
            shared,
            key=lambda key: (
                key_complexity(key),
                key[0],
                key[1],
                key[2],
            )
        )

        chosen = sorted_shared[0]

        learned_group_keys[group] = chosen

        group_debug[group] = {
            "examples": group_examples[group],
            "shared_exact_count": len(sorted_shared),
            "chosen": chosen,
            "shared_exact": sorted_shared,
        }

    rule = {
        "family": "mask_repair_rule",
        "rule_type": "learned_shared_predelete_transform_by_edge_group",
        "mask_color": MASK_COLOR,
        "learned_group_keys": learned_group_keys,
        "group_debug": group_debug,
        "pair_results": pair_results,
        "train_pair_count": len(train_pairs),
    }

    train_exact = 0
    train_score = 0

    for pair in train_pairs:
        predicted = apply_mask_repair_rule(rule, pair["input"])
        score, exact = score_prediction(predicted, pair["output"])

        train_score += score

        if exact:
            train_exact += 1

    if train_exact != len(train_pairs):
        return None

    rule["train_exact_count"] = train_exact
    rule["train_score"] = train_score

    return rule


# ============================================================
# APPLY RULE
# ============================================================

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


def apply_mask_repair_rule(rule, input_grid):
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

    predicted = apply_candidate_to_input(
        learned_candidate,
        input_grid,
        bbox,
    )

    if prediction_is_valid(predicted):
        return predicted

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
# ROUTER-FRIENDLY SCORER
# ============================================================

def score_mask_repair_rule(train_pairs):
    rule = learn_mask_repair_rule(train_pairs)

    if rule is None:
        return None

    return {
        "family": "mask_repair_rule",
        "rule": rule,
        "score": rule["train_score"],
        "exact_count": rule["train_exact_count"],
        "train_pair_count": rule["train_pair_count"],
    }


def solve_mask_repair(train_pairs, test_input):
    rule = learn_mask_repair_rule(train_pairs)

    if rule is None:
        return None

    return apply_mask_repair_rule(rule, test_input)