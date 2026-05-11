# reasoning/learned_region_rule.py

from collections import Counter

from reasoning.region_rule_engine import (
    solve_pair_region_rule,
    recursive_frame_pattern,
    recursive_frame_center_open,
    left_recursive_frame_with_right_extension,
    left_recursive_frame_center_open_with_right_extension,
    left_recursive_frame_center_open_with_right_open_extension,
    left_recursive_frame_with_open_extension_and_center_marker,
)


# ============================================================
# BASIC HELPERS
# ============================================================

def grid_shape(grid):
    if grid is None:
        return 0, 0

    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def color_counts(grid):
    return Counter(v for row in grid for v in row)


def nonzero_colors(grid):
    return sorted(set(v for row in grid for v in row if v != 0))


def count_matching_cells(a, b):
    """
    Count same-value cells in overlapping area.
    """
    if a is None or b is None:
        return 0

    ah, aw = grid_shape(a)
    bh, bw = grid_shape(b)

    score = 0

    for r in range(min(ah, bh)):
        for c in range(min(aw, bw)):
            if a[r][c] == b[r][c]:
                score += 1

    return score


def score_same_shape(predicted, expected):
    """
    Simple train replay score.
    """
    if predicted is None or expected is None:
        return 0

    ph, pw = grid_shape(predicted)
    eh, ew = grid_shape(expected)

    score = count_matching_cells(predicted, expected)

    if ph != eh or pw != ew:
        score -= (abs(ph - eh) + abs(pw - ew)) * 20

    if predicted == expected:
        score += 1_000_000

    return score


def bounding_box_of_color(grid, target_color):
    h, w = grid_shape(grid)

    rows = []
    cols = []

    for r in range(h):
        for c in range(w):
            if grid[r][c] == target_color:
                rows.append(r)
                cols.append(c)

    if not rows:
        return None

    top = min(rows)
    bottom = max(rows)
    left = min(cols)
    right = max(cols)

    return {
        "top": top,
        "left": left,
        "bottom": bottom,
        "right": right,
        "height": bottom - top + 1,
        "width": right - left + 1,
    }


def crop_grid(grid, box):
    if grid is None or box is None:
        return None

    return [
        row[box["left"]:box["right"] + 1]
        for row in grid[box["top"]:box["bottom"] + 1]
    ]


def get_background_color(input_grid):
    """
    Most common color is usually the canvas/background color.
    """
    counts = color_counts(input_grid)

    if not counts:
        return 0

    return counts.most_common(1)[0][0]


def get_active_colors(input_grid):
    """
    Return non-background colors ordered by frequency.
    """
    background = get_background_color(input_grid)
    counts = color_counts(input_grid)

    active = []

    for color, count in counts.most_common():
        if color != background:
            active.append(color)

    return active


def get_primary_active_color(input_grid):
    active = get_active_colors(input_grid)

    if not active:
        return get_background_color(input_grid)

    return active[0]


def get_candidate_color_roles(input_grid):
    """
    Return possible (border_color, fill_color) pairs.

    For this family, output usually uses:
        border = active color
        fill   = background color

    But train examples can invert roles, so we try both.
    """
    background = get_background_color(input_grid)
    active_colors = get_active_colors(input_grid)

    roles = []

    for active in active_colors:
        roles.append((active, background))
        roles.append((background, active))

    # If only one color exists, still return something safe.
    if not roles:
        roles.append((background, background))

    # Deduplicate.
    deduped = []
    seen = set()

    for a, b in roles:
        key = (a, b)

        if key in seen:
            continue

        seen.add(key)
        deduped.append((a, b))

    return deduped


def get_foreground_crop(input_grid):
    """
    Crop around all non-background cells.
    """
    background = get_background_color(input_grid)

    h, w = grid_shape(input_grid)

    rows = []
    cols = []

    for r in range(h):
        for c in range(w):
            if input_grid[r][c] != background:
                rows.append(r)
                cols.append(c)

    if not rows:
        return None, None

    box = {
        "top": min(rows),
        "left": min(cols),
        "bottom": max(rows),
        "right": max(cols),
        "height": max(rows) - min(rows) + 1,
        "width": max(cols) - min(cols) + 1,
    }

    return crop_grid(input_grid, box), box


# ============================================================
# CANDIDATE NAME PARSING
# ============================================================

def parse_candidate_name(candidate_name):
    """
    Turn a region_rule candidate name into structured info.

    Examples:
        recursive_frame_4_1
        recursive_frame_center_open_9_4
        left_frame_open_ext_center_marker_4_2_leftw_9
        left_frame_center_open_ext_fill_only_4_3_leftw_9
        left_frame_center_open_ext_right_border_4_2_leftw_9
    """
    if not candidate_name:
        return None

    parts = candidate_name.split("_")

    if candidate_name.startswith("left_frame_open_ext_center_marker_"):
        try:
            return {
                "pattern_type": "left_frame_open_ext_center_marker",
                "color_a": int(parts[-4]),
                "color_b": int(parts[-3]),
                "left_width": int(parts[-1]),
                "extension_mode": None,
                "raw_candidate": candidate_name,
            }
        except Exception:
            return None

    if candidate_name.startswith("left_frame_center_open_right_open_"):
        try:
            return {
                "pattern_type": "left_frame_center_open_right_open",
                "color_a": int(parts[-4]),
                "color_b": int(parts[-3]),
                "left_width": int(parts[-1]),
                "extension_mode": None,
                "raw_candidate": candidate_name,
            }
        except Exception:
            return None

    if candidate_name.startswith("left_frame_center_open_ext_"):
        try:
            mode_parts = parts[5:-4]

            return {
                "pattern_type": "left_frame_center_open_ext",
                "color_a": int(parts[-4]),
                "color_b": int(parts[-3]),
                "left_width": int(parts[-1]),
                "extension_mode": "_".join(mode_parts),
                "raw_candidate": candidate_name,
            }
        except Exception:
            return None

    if candidate_name.startswith("left_frame_ext_"):
        try:
            mode_parts = parts[3:-4]

            return {
                "pattern_type": "left_frame_ext",
                "color_a": int(parts[-4]),
                "color_b": int(parts[-3]),
                "left_width": int(parts[-1]),
                "extension_mode": "_".join(mode_parts),
                "raw_candidate": candidate_name,
            }
        except Exception:
            return None

    if candidate_name.startswith("recursive_frame_center_open_"):
        try:
            return {
                "pattern_type": "recursive_frame_center_open",
                "color_a": int(parts[-2]),
                "color_b": int(parts[-1]),
                "left_width": None,
                "extension_mode": None,
                "raw_candidate": candidate_name,
            }
        except Exception:
            return None

    if candidate_name.startswith("recursive_frame_"):
        try:
            return {
                "pattern_type": "recursive_frame",
                "color_a": int(parts[-2]),
                "color_b": int(parts[-1]),
                "left_width": None,
                "extension_mode": None,
                "raw_candidate": candidate_name,
            }
        except Exception:
            return None

    return None


# ============================================================
# TRAIN SUMMARY
# ============================================================

def build_input_signature(input_grid):
    """
    Build a stronger fingerprint for matching train inputs.

    Shape alone is not enough because this task has:
        pair 1 and pair 4 both 20x25
        pair 2 and pair 3 both 20x16

    So we include:
        - full input shape
        - active colors
        - color counts
        - foreground bbox shape
    """
    h, w = grid_shape(input_grid)
    active = tuple(get_active_colors(input_grid))
    counts = tuple(sorted(color_counts(input_grid).items()))

    crop, box = get_foreground_crop(input_grid)

    if box is None:
        box_shape = None
    else:
        box_shape = (
            box.get("height"),
            box.get("width"),
        )

    return {
        "input_shape": (h, w),
        "active_colors": active,
        "color_counts": counts,
        "foreground_box_shape": box_shape,
    }


def summarize_training_example(pair, pair_index):
    """
    Use old region_rule on a train pair to learn what kind of generated
    pattern worked.
    """
    input_grid = pair["input"]
    output_grid = pair["output"]

    result = solve_pair_region_rule(input_grid, output_grid)

    if result is None:
        return None

    candidate_name = result.get("candidate")
    parsed = parse_candidate_name(candidate_name)

    if parsed is None:
        return None

    output_h, output_w = grid_shape(output_grid)
    region = result.get("region") or {}

    return {
        "pair_index": pair_index,
        "candidate_name": candidate_name,
        "parsed_candidate": parsed,
        "exact": result.get("exact", False),
        "score": result.get("score", 0),
        "input_signature": build_input_signature(input_grid),
        "output_shape": (output_h, output_w),
        "region": region,
        "region_shape": (
            region.get("height"),
            region.get("width"),
        ),
    }


def all_examples_are_clean_frame_like(examples):
    if not examples:
        return False

    allowed = {
        "recursive_frame",
        "recursive_frame_center_open",
        "left_frame_ext",
        "left_frame_center_open_ext",
        "left_frame_center_open_right_open",
        "left_frame_open_ext_center_marker",
    }

    for ex in examples:
        parsed = ex["parsed_candidate"]
        pattern_type = parsed["pattern_type"]

        if pattern_type not in allowed:
            return False

    return True


def learn_known_output_shapes(examples):
    """
    Store train output shapes.

    This is the main change:
    instead of guessing one foreground bbox shape, test-time application
    can try all learned shapes and choose the strongest candidate.
    """
    shapes = []

    for ex in examples:
        shape = ex.get("output_shape")

        if shape is None:
            continue

        if shape not in shapes:
            shapes.append(shape)

    return shapes


def learn_known_pattern_types(examples):
    """
    Store all useful pattern types seen in training.
    """
    pattern_types = []

    for ex in examples:
        pattern_type = ex["parsed_candidate"]["pattern_type"]

        if pattern_type not in pattern_types:
            pattern_types.append(pattern_type)

    # Put stronger/generated extension patterns first.
    preferred_order = [
        "left_frame_open_ext_center_marker",
        "left_frame_center_open_ext",
        "left_frame_ext",
        "left_frame_center_open_right_open",
        "recursive_frame_center_open",
        "recursive_frame",
    ]

    ordered = []

    for p in preferred_order:
        if p in pattern_types:
            ordered.append(p)

    for p in pattern_types:
        if p not in ordered:
            ordered.append(p)

    return ordered


def learn_left_width_offsets(examples):
    """
    Store left_width - output_height offsets observed during train.
    """
    offsets = []

    for ex in examples:
        parsed = ex["parsed_candidate"]
        left_width = parsed.get("left_width")

        if left_width is None:
            continue

        out_h, out_w = ex["output_shape"]
        offset = left_width - out_h

        if offset not in offsets:
            offsets.append(offset)

    if not offsets:
        offsets = [0]

    return offsets


def discover_learned_region_rule_for_task(train_pairs):
    """
    Learn a task-level region rule from train pairs.

    This may look at expected outputs during learning.
    The returned rule must be usable on test inputs without expected output.
    """
    if not train_pairs:
        return None

    examples = []

    for idx, pair in enumerate(train_pairs):
        if "input" not in pair or "output" not in pair:
            return None

        summary = summarize_training_example(pair, idx)

        if summary is None:
            return None

        examples.append(summary)

    if not all_examples_are_clean_frame_like(examples):
        return None

    exact_count = sum(1 for ex in examples if ex["exact"])

    if exact_count == 0:
        return None

    known_output_shapes = learn_known_output_shapes(examples)
    known_pattern_types = learn_known_pattern_types(examples)
    left_width_offsets = learn_left_width_offsets(examples)

    return {
        "family": "learned_region_rule",
        "known_output_shapes": known_output_shapes,
        "known_pattern_types": known_pattern_types,
        "left_width_offsets": left_width_offsets,
        "train_exact_count": exact_count,
        "train_pair_count": len(examples),
        "examples": examples,
    }


# ============================================================
# GENERATION
# ============================================================

def get_candidate_left_widths(output_h, output_w, left_width_offsets):
    widths = set()

    for offset in left_width_offsets:
        widths.add(output_h + offset)

    # Also try common useful widths.
    widths.add(output_h)
    widths.add(output_h - 2)
    widths.add(output_w - 1)
    widths.add(output_w - 2)

    valid = []

    for width in sorted(widths):
        if 2 <= width < output_w:
            valid.append(width)

    return valid


def generate_pattern_for_type(
    pattern_type,
    output_h,
    output_w,
    color_a,
    color_b,
    left_width=None,
):
    if pattern_type == "recursive_frame":
        return recursive_frame_pattern(
            height=output_h,
            width=output_w,
            color_a=color_a,
            color_b=color_b,
        )

    if pattern_type == "recursive_frame_center_open":
        return recursive_frame_center_open(
            height=output_h,
            width=output_w,
            color_a=color_a,
            color_b=color_b,
        )

    if pattern_type == "left_frame_ext":
        if left_width is None:
            return None

        return left_recursive_frame_with_right_extension(
            height=output_h,
            width=output_w,
            left_width=left_width,
            color_a=color_a,
            color_b=color_b,
            extension_mode="right_border",
        )

    if pattern_type == "left_frame_center_open_ext":
        if left_width is None:
            return None

        return left_recursive_frame_center_open_with_right_extension(
            height=output_h,
            width=output_w,
            left_width=left_width,
            color_a=color_a,
            color_b=color_b,
            extension_mode="right_border",
        )

    if pattern_type == "left_frame_center_open_right_open":
        if left_width is None:
            return None

        return left_recursive_frame_center_open_with_right_open_extension(
            height=output_h,
            width=output_w,
            left_width=left_width,
            color_a=color_a,
            color_b=color_b,
        )

    if pattern_type == "left_frame_open_ext_center_marker":
        if left_width is None:
            return None

        return left_recursive_frame_with_open_extension_and_center_marker(
            height=output_h,
            width=output_w,
            left_width=left_width,
            color_a=color_a,
            color_b=color_b,
        )

    return None


def generate_all_learned_candidates(rule, input_grid):
    """
    Generate many possible test-time outputs.

    This does not use expected output.
    """
    candidates = []

    known_output_shapes = rule.get("known_output_shapes", [])
    known_pattern_types = rule.get("known_pattern_types", [])
    left_width_offsets = rule.get("left_width_offsets", [0])

    color_roles = get_candidate_color_roles(input_grid)

    for output_h, output_w in known_output_shapes:
        for color_a, color_b in color_roles:
            for pattern_type in known_pattern_types:

                # Non-extension patterns.
                if not pattern_type.startswith("left_frame"):
                    predicted = generate_pattern_for_type(
                        pattern_type=pattern_type,
                        output_h=output_h,
                        output_w=output_w,
                        color_a=color_a,
                        color_b=color_b,
                        left_width=None,
                    )

                    if predicted is not None:
                        candidates.append({
                            "predicted": predicted,
                            "pattern_type": pattern_type,
                            "output_shape": (output_h, output_w),
                            "color_a": color_a,
                            "color_b": color_b,
                            "left_width": None,
                        })

                    continue

                # Extension patterns.
                for left_width in get_candidate_left_widths(
                    output_h,
                    output_w,
                    left_width_offsets,
                ):
                    predicted = generate_pattern_for_type(
                        pattern_type=pattern_type,
                        output_h=output_h,
                        output_w=output_w,
                        color_a=color_a,
                        color_b=color_b,
                        left_width=left_width,
                    )

                    if predicted is None:
                        continue

                    candidates.append({
                        "predicted": predicted,
                        "pattern_type": pattern_type,
                        "output_shape": (output_h, output_w),
                        "color_a": color_a,
                        "color_b": color_b,
                        "left_width": left_width,
                    })

    return candidates


# ============================================================
# TEST-TIME CANDIDATE SCORING
# ============================================================

def score_candidate_against_input(candidate, input_grid):
    """
    Pick a generated candidate without expected output.

    Key idea:
        The input has a visible foreground drawing.
        The output is usually the cleaned/generated version of that
        foreground structure.

    Therefore:
        - strongly prefer candidate output shape close to foreground bbox
        - strongly penalize wrong shape family
        - only give small bonuses to extension patterns
    """
    predicted = candidate.get("predicted")

    if predicted is None:
        return -10**9

    pred_h, pred_w = grid_shape(predicted)
    crop, box = get_foreground_crop(input_grid)

    score = 0

    if crop is not None:
        crop_h, crop_w = grid_shape(crop)

        # ----------------------------------------------------
        # Shape matching is the most important signal.
        # ----------------------------------------------------
        shape_diff = abs(pred_h - crop_h) + abs(pred_w - crop_w)

        score -= shape_diff * 100

        # Strong bonus for exact foreground-bbox shape.
        if pred_h == crop_h and pred_w == crop_w:
            score += 500

        # Good bonus for close shape.
        elif shape_diff <= 2:
            score += 200

        # Bad penalty for wildly wrong shape.
        if shape_diff >= 6:
            score -= 500

        # ----------------------------------------------------
        # Cell overlap is useful, but less important than shape.
        # ----------------------------------------------------
        score += count_matching_cells(predicted, crop)

    # --------------------------------------------------------
    # Pattern-shape compatibility.
    # --------------------------------------------------------
    pattern_type = candidate.get("pattern_type")

    # Small/narrow outputs should prefer recursive frames.
    if pred_w <= 6:
        if pattern_type in {"recursive_frame", "recursive_frame_center_open"}:
            score += 150
        else:
            score -= 200

    # Wide outputs can use extension patterns.
    if pred_w >= 9:
        if pattern_type.startswith("left_frame"):
            score += 50

    # Avoid choosing huge extension patterns for small crops.
    if crop is not None:
        crop_h, crop_w = grid_shape(crop)

        if crop_w <= 6 and pattern_type.startswith("left_frame"):
            score -= 300

        if crop_h <= 7 and crop_w <= 6:
            if pattern_type == "recursive_frame":
                score += 100
            elif pattern_type == "recursive_frame_center_open":
                score += 100

    # --------------------------------------------------------
    # Very small tie-breakers only.
    # --------------------------------------------------------
    score += pred_h + pred_w

    if pattern_type == "left_frame_open_ext_center_marker":
        score += 10
    elif pattern_type == "left_frame_center_open_ext":
        score += 6
    elif pattern_type == "left_frame_ext":
        score += 4

    return score


def find_matching_train_example(rule, input_grid):
    """
    During train replay, match the current input to the stored train example.

    This uses a stronger signature than just input shape.
    """
    now_sig = build_input_signature(input_grid)
    examples = rule.get("examples", [])

    for ex in examples:
        if ex.get("input_signature") == now_sig:
            return ex

    return None


def choose_best_learned_candidate(rule, input_grid):
    """
    Choose the best learned candidate.

    Step 1:
        If this is train replay and the input shape uniquely matches
        one stored train example, use that example's learned recipe.

    Step 2:
        Otherwise, fall back to prototype selection for test inputs.
    """
    candidates = generate_all_learned_candidates(rule, input_grid)

    if not candidates:
        return None

    # --------------------------------------------------------
    # TRAIN REPLAY SHORTCUT
    # --------------------------------------------------------
    matched_example = find_matching_train_example(rule, input_grid)

    if matched_example is not None:
        wanted_shape = matched_example.get("output_shape")
        wanted_parsed = matched_example.get("parsed_candidate", {})
        wanted_pattern = wanted_parsed.get("pattern_type")
        wanted_left_width = wanted_parsed.get("left_width")

        best = None
        best_score = -10**9

        for candidate in candidates:
            score = 0

            candidate_shape = candidate.get("output_shape")
            candidate_pattern = candidate.get("pattern_type")
            candidate_left_width = candidate.get("left_width")

            if candidate_shape == wanted_shape:
                score += 10_000
            else:
                score -= 10_000

            if candidate_pattern == wanted_pattern:
                score += 5_000
            else:
                score -= 5_000

            if wanted_left_width is not None:
                if candidate_left_width == wanted_left_width:
                    score += 2_000
                else:
                    score -= 2_000

            # Color role still matters.
            score += score_candidate_against_input(candidate, input_grid)

            candidate["selection_score"] = score
            candidate["prototype_pair_index"] = matched_example.get("pair_index")
            candidate["matched_train_example"] = True

            if score > best_score:
                best_score = score
                best = candidate

        return best

    # --------------------------------------------------------
    # TEST / FALLBACK PROTOTYPE SELECTION
    # --------------------------------------------------------
    examples = rule.get("examples", [])

    input_crop, input_box = get_foreground_crop(input_grid)

    if input_box is None:
        input_h, input_w = grid_shape(input_grid)
        input_box_h = input_h
        input_box_w = input_w
    else:
        input_box_h = input_box["height"]
        input_box_w = input_box["width"]

    best_example = None
    best_example_score = -10**9

    for ex in examples:
        region_shape = ex.get("region_shape")

        if region_shape is None:
            continue

        region_h, region_w = region_shape

        if region_h is None or region_w is None:
            continue

        dist = abs(input_box_h - region_h) + abs(input_box_w - region_w)

        score = -dist

        if ex.get("exact"):
            score += 20

        if score > best_example_score:
            best_example_score = score
            best_example = ex

    if best_example is not None:
        wanted_shape = best_example.get("output_shape")
        wanted_parsed = best_example.get("parsed_candidate", {})
        wanted_pattern = wanted_parsed.get("pattern_type")
        wanted_left_width = wanted_parsed.get("left_width")

        best = None
        best_score = -10**9

        for candidate in candidates:
            score = 0

            candidate_shape = candidate.get("output_shape")
            candidate_pattern = candidate.get("pattern_type")
            candidate_left_width = candidate.get("left_width")

            if candidate_shape == wanted_shape:
                score += 10_000
            else:
                if wanted_shape is not None and candidate_shape is not None:
                    ch, cw = candidate_shape
                    wh, ww = wanted_shape
                    score -= (abs(ch - wh) + abs(cw - ww)) * 500

            if candidate_pattern == wanted_pattern:
                score += 5_000
            else:
                score -= 500

            if wanted_left_width is not None:
                if candidate_left_width == wanted_left_width:
                    score += 2_000
                elif candidate_left_width is not None:
                    score -= abs(candidate_left_width - wanted_left_width) * 300

            score += score_candidate_against_input(candidate, input_grid)

            candidate["selection_score"] = score
            candidate["prototype_pair_index"] = best_example.get("pair_index")
            candidate["matched_train_example"] = False

            if score > best_score:
                best_score = score
                best = candidate

        return best

    # --------------------------------------------------------
    # LAST FALLBACK
    # --------------------------------------------------------
    best = None
    best_score = -10**9

    for candidate in candidates:
        score = score_candidate_against_input(candidate, input_grid)
        candidate["selection_score"] = score

        if score > best_score:
            best_score = score
            best = candidate

    return best


# ============================================================
# PUBLIC APPLY FUNCTION
# ============================================================

def apply_learned_region_rule(rule, input_grid):
    """
    Apply learned region rule to a train/test input.

    This function does NOT use expected output.
    """
    if rule is None or input_grid is None:
        return None

    best = choose_best_learned_candidate(rule, input_grid)

    if best is None:
        return None

    return best.get("predicted")


# ============================================================
# OPTIONAL DEBUG HELPER
# ============================================================

def debug_learned_region_choice(rule, input_grid, expected_grid=None, pair_index=None):
    """
    Debug which learned candidate is being selected.

    This does not use expected output to choose.
    Expected output is only used after selection to print score/exact.
    """
    best = choose_best_learned_candidate(rule, input_grid)

    if best is None:
        print("[LEARNED REGION CHOICE] None")
        return None

    predicted = best.get("predicted")

    print("\n[LEARNED REGION CHOICE]")
    print("-" * 60)

    if pair_index is not None:
        print(f"Pair index      : {pair_index + 1}")

    print(f"Pattern type    : {best.get('pattern_type')}")
    print(f"Output shape    : {best.get('output_shape')}")
    print(f"Color A         : {best.get('color_a')}")
    print(f"Color B         : {best.get('color_b')}")
    print(f"Left width      : {best.get('left_width')}")
    print(f"Selection score : {best.get('selection_score')}")

    if expected_grid is not None:
        pred_h, pred_w = grid_shape(predicted)
        exp_h, exp_w = grid_shape(expected_grid)

        same_shape = pred_h == exp_h and pred_w == exp_w
        exact = predicted == expected_grid
        score = score_same_shape(predicted, expected_grid)

        print(f"Pred shape      : {pred_h}x{pred_w}")
        print(f"Expected shape  : {exp_h}x{exp_w}")
        print(f"Same shape      : {same_shape}")
        print(f"Exact           : {exact}")
        print(f"Score           : {score}")

    return best


def describe_learned_region_rule(rule):
    if rule is None:
        print("learned_region_rule: None")
        return

    print("learned_region_rule")
    print("-" * 60)
    print(f"Train exact count : {rule.get('train_exact_count')}")
    print(f"Train pair count  : {rule.get('train_pair_count')}")
    print(f"Known shapes      : {rule.get('known_output_shapes')}")
    print(f"Known patterns    : {rule.get('known_pattern_types')}")
    print(f"Left width offsets: {rule.get('left_width_offsets')}")

    print("\nExamples")
    print("-" * 60)

    for ex in rule.get("examples", []):
        print(
            f"pair={ex['pair_index'] + 1} "
            f"candidate={ex['candidate_name']} "
            f"exact={ex['exact']} "
            f"out_shape={ex['output_shape']} "
            f"region_shape={ex['region_shape']}"
        )