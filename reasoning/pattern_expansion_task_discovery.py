from reasoning.pattern_expansion_rule import solve_pair_pattern_expansion
from reasoning.pattern_expansion_rule import apply_pattern_expansion_mode

# ============================================================
# GRID FEATURE EXTRACTION
# ============================================================
def grid_shape(grid):
    if grid is None:
        return 0, 0

    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def get_non_background_cells(grid):
    """
    Treat 0 as background.

    Returns:
        list of (row, col, value)
    """
    cells = []

    if grid is None:
        return cells

    h, w = grid_shape(grid)

    for r in range(h):
        for c in range(w):
            val = grid[r][c]
            if val != 0:
                cells.append((r, c, val))

    return cells


def extract_pattern_expansion_features(grid):
    """
    Extract simple geometry features from the input grid.

    These features are used to learn:
        input shape / layout -> expansion mode

    This is the first version.
    Later we can add stronger features like:
        - edge contact
        - line directions
        - color runs
        - symmetry
        - connected components
    """
    h, w = grid_shape(grid)
    cells = get_non_background_cells(grid)

    if not cells:
        return {
            "height": h,
            "width": w,
            "area": h * w,
            "nonzero_count": 0,
            "fill_ratio": 0,
            "bbox_h": 0,
            "bbox_w": 0,
            "bbox_area": 0,
            "bbox_fill_ratio": 0,
            "center_r": 0,
            "center_c": 0,
            "touches_top": False,
            "touches_bottom": False,
            "touches_left": False,
            "touches_right": False,
            "dominant_color": None,
            "color_count": 0,
        }

    rows = [r for r, c, v in cells]
    cols = [c for r, c, v in cells]
    vals = [v for r, c, v in cells]

    min_r = min(rows)
    max_r = max(rows)
    min_c = min(cols)
    max_c = max(cols)

    bbox_h = max_r - min_r + 1
    bbox_w = max_c - min_c + 1
    bbox_area = bbox_h * bbox_w

    color_counts = {}
    for v in vals:
        color_counts[v] = color_counts.get(v, 0) + 1

    dominant_color = max(color_counts, key=color_counts.get)

    return {
        "height": h,
        "width": w,
        "area": h * w,
        "nonzero_count": len(cells),
        "fill_ratio": len(cells) / (h * w) if h * w else 0,
        "bbox_h": bbox_h,
        "bbox_w": bbox_w,
        "bbox_area": bbox_area,
        "bbox_fill_ratio": len(cells) / bbox_area if bbox_area else 0,
        "center_r": (min_r + max_r) / 2,
        "center_c": (min_c + max_c) / 2,
        "touches_top": min_r == 0,
        "touches_bottom": max_r == h - 1,
        "touches_left": min_c == 0,
        "touches_right": max_c == w - 1,
        "dominant_color": dominant_color,
        "color_count": len(color_counts),
    }


# ============================================================
# FEATURE DISTANCE
# ============================================================
def feature_distance(a, b):
    """
    Compare two feature dictionaries.

    Lower score means the inputs are more similar.

    This is how the test input chooses which learned train example
    it most resembles.
    """
    numeric_keys = [
        "height",
        "width",
        "area",
        "nonzero_count",
        "fill_ratio",
        "bbox_h",
        "bbox_w",
        "bbox_area",
        "bbox_fill_ratio",
        "center_r",
        "center_c",
        "color_count",
    ]

    bool_keys = [
        "touches_top",
        "touches_bottom",
        "touches_left",
        "touches_right",
    ]

    dist = 0

    for key in numeric_keys:
        av = a.get(key, 0)
        bv = b.get(key, 0)
        dist += abs(av - bv)

    for key in bool_keys:
        if a.get(key) != b.get(key):
            dist += 5

    if a.get("dominant_color") != b.get("dominant_color"):
        dist += 10

    return dist


# ============================================================
# LEARN TASK-LEVEL MODE MAPPING
# ============================================================
def discover_pattern_expansion_rule_for_task(train_pairs):
    """
    Learn a task-level pattern expansion rule.

    Important:
    This does NOT assume one fixed mode explains every train pair.

    Instead it learns:

        train input features -> best expansion mode

    Example:
        Pair 1 features -> shifted_row_3_col_-2
        Pair 2 features -> shifted_row_3_col_-4
        Pair 3 features -> shifted_row_2_col_-1

    Then at test time:
        extract test features
        find closest train feature
        reuse that train pair's learned mode
    """
    if not train_pairs:
        return None

    learned_examples = []
    total_score = 0
    exact_count = 0
    pair_results = []

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        output_grid = pair["output"]

        features = extract_pattern_expansion_features(input_grid)

        result = solve_pair_pattern_expansion(input_grid, output_grid)

        if result is None:
            learned_examples.append({
                "pair_index": pair_index,
                "features": features,
                "mode": None,
                "score": -100000,
                "exact": False,
            })

            pair_results.append({
                "pair_index": pair_index,
                "mode": None,
                "score": -100000,
                "exact": False,
                "features": features,
            })

            total_score -= 100000
            continue

        mode = result.get("mode", "unknown")
        score = result.get("adjusted_score", result.get("score", 0))
        exact = result.get("exact", False)

        learned_examples.append({
            "pair_index": pair_index,
            "features": features,
            "mode": mode,
            "score": score,
            "exact": exact,
        })

        pair_results.append({
            "pair_index": pair_index,
            "mode": mode,
            "score": score,
            "exact": exact,
            "features": features,
        })

        total_score += score

        if exact:
            exact_count += 1

    valid_modes = [
        ex["mode"]
        for ex in learned_examples
        if ex["mode"] is not None
    ]

    if not valid_modes:
        return None

    mode_counts = {}
    mode_scores = {}

    for ex in learned_examples:
        mode = ex["mode"]

        if mode is None:
            continue

        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        mode_scores[mode] = mode_scores.get(mode, 0) + ex["score"]

    best_overall_mode = sorted(
        mode_scores.keys(),
        key=lambda m: (
            mode_counts.get(m, 0),
            mode_scores.get(m, 0),
        ),
        reverse=True,
    )[0]

    return {
        "family": "pattern_expansion_feature_mapping_rule",
        "rule_type": "feature_to_mode_mapping",
        "default_mode": best_overall_mode,
        "total_score": total_score,
        "exact_count": exact_count,
        "pair_count": len(train_pairs),
        "learned_examples": learned_examples,
        "results": pair_results,
        "mode_counts": mode_counts,
        "mode_scores": mode_scores,
    }


# ============================================================
# CHOOSE MODE FOR TEST INPUT
# ============================================================
def choose_mode_for_test_input(rule, test_input):
    """
    Pick the expansion mode for a test input.

    It compares the test input features to every train input feature,
    then chooses the mode from the closest train example.
    """
    if rule is None:
        return None

    test_features = extract_pattern_expansion_features(test_input)

    learned_examples = rule.get("learned_examples", [])

    best_example = None
    best_distance = None

    for ex in learned_examples:
        mode = ex.get("mode")

        if mode is None:
            continue

        train_features = ex.get("features", {})
        dist = feature_distance(test_features, train_features)

        if best_distance is None or dist < best_distance:
            best_distance = dist
            best_example = ex

    if best_example is None:
        return rule.get("default_mode")

    return best_example.get("mode")


# ============================================================
# APPLY TASK-LEVEL RULE TO TEST
# ============================================================
def apply_pattern_expansion_task_rule(rule, test_input):
    """
    Apply the learned feature -> mode mapping to a test input.

    Current limitation:
    Your existing pattern_expansion_rule.py does not yet expose a function
    for forcing a specific mode by name.

    So this function:
    1. Chooses the intended mode using the learned mapping.
    2. Calls the existing solver as fallback.
    3. Attaches the selected mode for debugging if possible.

    Next upgrade:
    Add a function in pattern_expansion_rule.py like:

        apply_pattern_expansion_mode(input_grid, mode, output_shape=(20, 20))

    Then this file can force the selected mode exactly.
    """
    if rule is None:
        return None

    selected_mode = choose_mode_for_test_input(rule, test_input)



    selected_mode = choose_mode_for_test_input(rule, test_input)
    prediction = apply_pattern_expansion_mode(
        test_input,
        selected_mode,
        out_h=20,
        out_w=20,
    )
    return prediction

    # Save debug info into the result dictionary if needed later.
    result["task_selected_mode"] = selected_mode
    result["task_rule_family"] = rule.get("family")

    return prediction