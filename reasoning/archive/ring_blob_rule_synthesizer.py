# reasoning/ring_blob_rule_synthesizer.py
"""
Ring / Blob Rule Synthesizer

Goal:
    Learn a task-level rule from all train pairs.

Important:
    This file should NOT hardcode task ids.

    It should learn from records like:

        input scene signature -> output shape/template/colors

    Then apply that learned rule to a hidden/test input.

Current focus:
    Ring/blob visual abstraction tasks such as 2d0172a1.

Main public functions used by router / debug tools:
    learn_ring_blob_rule_synthesizer(train_pairs)
    predict_with_ring_blob_rule_synthesizer(rules, input_grid)
    scene_signature(scene)
    print_rule_synthesizer_summary(rules)
    print_rule_synthesizer_prediction_summary(result)
"""

from collections import Counter


# ============================================================
# Imports
# ============================================================

try:
    from reasoning.archive.ring_blob_scene import learn_ring_blob_scene
except ImportError:
    from ring_blob_scene import learn_ring_blob_scene


# ============================================================
# Basic grid helpers
# ============================================================

def grid_shape(grid):
    if grid is None:
        return 0, 0

    if not grid:
        return 0, 0

    return len(grid), len(grid[0])


def copy_grid(grid):
    if grid is None:
        return None

    return [row[:] for row in grid]


def color_counts(grid):
    counts = Counter()

    if grid is None:
        return counts

    for row in grid:
        for value in row:
            counts[value] += 1

    return counts


def most_common_color(grid):
    counts = color_counts(grid)

    if not counts:
        return 0

    return counts.most_common(1)[0][0]


def unique_in_order(items):
    seen = set()
    result = []

    for item in items:
        key = repr(item)

        if key not in seen:
            seen.add(key)
            result.append(item)

    return result


def safe_tuple(value):
    if value is None:
        return tuple()

    if isinstance(value, tuple):
        return value

    if isinstance(value, list):
        return tuple(value)

    return tuple()


# ============================================================
# Color helpers
# ============================================================

def get_input_background_color(input_grid):
    return most_common_color(input_grid)


def get_input_foreground_color(input_grid):
    """
    Pick the most common non-background color from the input.
    """
    background = get_input_background_color(input_grid)
    counts = color_counts(input_grid)

    if background in counts:
        del counts[background]

    if not counts:
        return background

    return counts.most_common(1)[0][0]


def get_output_fill_color(output_grid):
    """
    Output fill/background is usually the most common color.
    """
    return most_common_color(output_grid)


def get_output_foreground_from_edges(output_grid):
    """
    Output foreground/border is often dominant on the outer edge.
    """
    if output_grid is None:
        return 0

    h, w = grid_shape(output_grid)

    if h == 0 or w == 0:
        return 0

    counts = Counter()

    for c in range(w):
        counts[output_grid[0][c]] += 1
        counts[output_grid[h - 1][c]] += 1

    for r in range(h):
        counts[output_grid[r][0]] += 1
        counts[output_grid[r][w - 1]] += 1

    if not counts:
        return 0

    return counts.most_common(1)[0][0]


def get_output_colors(output_grid):
    counts = color_counts(output_grid)

    return sorted(counts.keys())


def get_output_fill_color_from_records(rules):
    """
    Find the most common output/background color from learned records.

    Used when resize_grid_to_shape needs to pad a prediction.
    """
    if rules is None:
        return None

    records = rules.get("records", [])

    counts = Counter()

    for record in records:
        output_grid = record.get("output_grid")

        if output_grid is None:
            output_grid = record.get("output")

        if output_grid is None:
            continue

        for row in output_grid:
            for value in row:
                counts[value] += 1

    if not counts:
        return None

    return counts.most_common(1)[0][0]


def recolor_grid(grid, old_fill, old_foreground, new_fill, new_foreground):
    """
    Recolor a copied train template into test colors.

    This is still template-based, but useful while we improve the
    learned drawing rule.
    """
    if grid is None:
        return None

    result = []

    for row in grid:
        new_row = []

        for value in row:
            if value == old_fill:
                new_row.append(new_fill)
            elif value == old_foreground:
                new_row.append(new_foreground)
            else:
                new_row.append(value)

        result.append(new_row)

    return result


# ============================================================
# Scene signature helpers
# ============================================================

def _scene_get(scene, key, default=None):
    if not isinstance(scene, dict):
        return default

    return scene.get(key, default)


def scene_ring_blob_counts(scene):
    """
    Extract the number of blobs assigned to each ring.

    Current ring_blob_scene.py stores this inside:

        scene["rings"][i]["blob_count"]

    Example:
        scene["rings"] = [
            {"blob_count": 1, "blob_layout": "single", ...},
            {"blob_count": 2, "blob_layout": "stacked_vertical", ...},
        ]

    So this should return:
        (1, 2)
    """
    if scene is None:
        return tuple()

    direct = _scene_get(scene, "ring_blob_counts")

    if direct is not None:
        return tuple(direct)

    # --------------------------------------------------------
    # This is the important current format.
    # --------------------------------------------------------
    rings = _scene_get(scene, "rings", [])

    if isinstance(rings, list):
        counts = []

        for ring in rings:
            if not isinstance(ring, dict):
                continue

            if "blob_count" in ring:
                counts.append(ring.get("blob_count", 0))
            elif "blobs" in ring and isinstance(ring["blobs"], list):
                counts.append(len(ring["blobs"]))
            else:
                counts.append(0)

        if counts:
            return tuple(counts)

    # --------------------------------------------------------
    # Older possible formats.
    # --------------------------------------------------------
    ring_items = _scene_get(scene, "ring_items")

    if isinstance(ring_items, list):
        counts = []

        for item in ring_items:
            if isinstance(item, dict):
                if "blob_count" in item:
                    counts.append(item.get("blob_count", 0))
                elif "blobs" in item and isinstance(item["blobs"], list):
                    counts.append(len(item["blobs"]))

        if counts:
            return tuple(counts)

    ring_summaries = _scene_get(scene, "ring_summaries")

    if isinstance(ring_summaries, list):
        counts = []

        for item in ring_summaries:
            if isinstance(item, dict):
                if "blob_count" in item:
                    counts.append(item.get("blob_count", 0))
                elif "blobs_inside" in item and isinstance(item["blobs_inside"], list):
                    counts.append(len(item["blobs_inside"]))

        if counts:
            return tuple(counts)

    return tuple()


def scene_ring_layouts(scene):
    """
    Extract ring layout labels.

    Current ring_blob_scene.py stores this inside:

        scene["rings"][i]["blob_layout"]

    Example:
        scene["rings"] = [
            {"blob_layout": "single"},
            {"blob_layout": "stacked_vertical"},
        ]

    So this should return:
        ("single", "stacked_vertical")
    """
    if scene is None:
        return tuple()

    direct = _scene_get(scene, "ring_layouts")

    if direct is not None:
        return tuple(direct)

    # --------------------------------------------------------
    # This is the important current format.
    # --------------------------------------------------------
    rings = _scene_get(scene, "rings", [])

    if isinstance(rings, list):
        layouts = []

        for ring in rings:
            if not isinstance(ring, dict):
                continue

            if "blob_layout" in ring:
                layouts.append(ring.get("blob_layout", "unknown"))
            elif "layout" in ring:
                layouts.append(ring.get("layout", "unknown"))
            elif "blob_count" in ring:
                count = ring.get("blob_count", 0)

                if count == 0:
                    layouts.append("none")
                elif count == 1:
                    layouts.append("single")
                else:
                    layouts.append(f"{count}_blobs")
            else:
                layouts.append("unknown")

        if layouts:
            return tuple(layouts)

    # --------------------------------------------------------
    # Older possible formats.
    # --------------------------------------------------------
    ring_items = _scene_get(scene, "ring_items")

    if isinstance(ring_items, list):
        layouts = []

        for item in ring_items:
            if isinstance(item, dict):
                layouts.append(item.get("blob_layout", item.get("layout", "unknown")))

        if layouts:
            return tuple(layouts)

    ring_summaries = _scene_get(scene, "ring_summaries")

    if isinstance(ring_summaries, list):
        layouts = []

        for item in ring_summaries:
            if isinstance(item, dict):
                layouts.append(item.get("blob_layout", item.get("layout", "unknown")))

        if layouts:
            return tuple(layouts)

    counts = scene_ring_blob_counts(scene)

    layouts = []

    for count in counts:
        if count == 0:
            layouts.append("none")
        elif count == 1:
            layouts.append("single")
        else:
            layouts.append(f"{count}_blobs")

    return tuple(layouts)


def scene_signature(scene):
    """
    Convert a ring/blob scene into a stable symbolic signature.

    This is what the shape learner and template chooser are allowed to use.
    """
    if scene is None:
        return {
            "ring_count": 0,
            "blob_count": 0,
            "outside_blob_count": 0,
            "ring_blob_counts": tuple(),
            "ring_layouts": tuple(),
        }

    ring_blob_counts = scene_ring_blob_counts(scene)
    ring_layouts = scene_ring_layouts(scene)

    ring_count = _scene_get(scene, "ring_count")

    if ring_count is None:
        rings = _scene_get(scene, "rings", [])
        ring_count = len(rings)

    blob_count = _scene_get(scene, "blob_count")

    if blob_count is None:
        blobs = _scene_get(scene, "blobs", [])
        blob_count = len(blobs)

    outside_blob_count = _scene_get(scene, "outside_blob_count")

    if outside_blob_count is None:
        outside_blobs = _scene_get(scene, "outside_blobs", [])

        if isinstance(outside_blobs, list):
            outside_blob_count = len(outside_blobs)
        else:
            inside_count = sum(ring_blob_counts)
            outside_blob_count = max(0, blob_count - inside_count)

    return {
        "ring_count": ring_count or 0,
        "blob_count": blob_count or 0,
        "outside_blob_count": outside_blob_count or 0,
        "ring_blob_counts": tuple(ring_blob_counts),
        "ring_layouts": tuple(ring_layouts),
    }


def signature_to_text(signature):
    if signature is None:
        return "None"

    return (
        f"rings={signature.get('ring_count')}, "
        f"blobs={signature.get('blob_count')}, "
        f"outside={signature.get('outside_blob_count')}, "
        f"ring_blob_counts={signature.get('ring_blob_counts')}, "
        f"layouts={signature.get('ring_layouts')}"
    )


def signatures_match(a, b):
    """
    Exact symbolic scene match.

    This is useful for normal all-train debugging, but leave-one-out
    prevents it from seeing the hidden pair.
    """
    if a is None or b is None:
        return False

    return (
        a.get("ring_count") == b.get("ring_count")
        and a.get("blob_count") == b.get("blob_count")
        and a.get("outside_blob_count") == b.get("outside_blob_count")
        and tuple(a.get("ring_blob_counts", [])) == tuple(b.get("ring_blob_counts", []))
        and tuple(a.get("ring_layouts", [])) == tuple(b.get("ring_layouts", []))
    )


def repeated_blob_ring_indexes(signature):
    """
    Return ring indexes where a ring contains more than one blob.
    """
    counts = signature.get("ring_blob_counts", [])

    result = []

    for index, count in enumerate(counts):
        if count >= 2:
            result.append(index)

    return tuple(result)


# ============================================================
# Output feature helpers
# ============================================================

def right_fill_lane_width(output_grid, fill_color):
    """
    Count fill-only columns on the right edge.
    """
    if output_grid is None:
        return 0

    h, w = grid_shape(output_grid)

    if h == 0 or w == 0:
        return 0

    lane_width = 0

    for c in range(w - 1, -1, -1):
        is_fill_col = True

        for r in range(h):
            if output_grid[r][c] != fill_color:
                is_fill_col = False
                break

        if is_fill_col:
            lane_width += 1
        else:
            break

    return lane_width


def output_record_for_pair(pair, index):
    """
    Convert one train pair into a learning record.

    This is the key data structure:

        scene signature -> output shape/template/colors
    """
    input_grid = pair["input"]
    output_grid = pair["output"]

    scene = learn_ring_blob_scene(input_grid)
    signature = scene_signature(scene)

    fill_color = get_output_fill_color(output_grid)
    foreground_color = get_output_foreground_from_edges(output_grid)

    return {
        "pair_index": index + 1,
        "input_grid": input_grid,
        "output_grid": output_grid,
        "scene": scene,
        "signature": signature,
        "output_shape": grid_shape(output_grid),
        "output_fill_color": fill_color,
        "output_foreground_color": foreground_color,
        "right_fill_lane_width": right_fill_lane_width(output_grid, fill_color),
    }


# ============================================================
# Learned symbolic output shape rule
# ============================================================

def get_signature_feature_values(signature):
    """
    Convert a ring/blob signature into numeric features.

    These are the facts the shape learner is allowed to use.

    No task id.
    No pair index.
    No hidden expected output.
    """
    if signature is None:
        signature = {}

    ring_blob_counts = signature.get("ring_blob_counts", [])

    inside_blob_count = 0

    for count in ring_blob_counts:
        inside_blob_count += count

    if ring_blob_counts:
        max_blobs_in_one_ring = max(ring_blob_counts)
    else:
        max_blobs_in_one_ring = 0

    ring_count = signature.get("ring_count", 0)
    blob_count = signature.get("blob_count", 0)
    outside_blob_count = signature.get("outside_blob_count", 0)

    return {
        "constant": 1,
        "ring_count": ring_count,
        "blob_count": blob_count,
        "outside_blob_count": outside_blob_count,
        "inside_blob_count": inside_blob_count,
        "max_blobs_in_one_ring": max_blobs_in_one_ring,

        # Useful derived features.
        "ring_plus_blob_count": ring_count + blob_count,
        "ring_plus_inside_blob_count": ring_count + inside_blob_count,
        "blob_plus_outside_blob_count": blob_count + outside_blob_count,
    }


def evaluate_linear_formula(formula, features):
    """
    Evaluate a simple formula.

    Formula format:
        {
            "constant": 3,
            "weights": {
                "blob_count": 2,
            }
        }

    Means:
        value = 3 + 2 * blob_count
    """
    value = formula.get("constant", 0)

    weights = formula.get("weights", {})

    for feature_name, weight in weights.items():
        value += weight * features.get(feature_name, 0)

    return value


def generate_candidate_formulas(feature_names):
    """
    Generate small formulas and let the train records choose.

    Important:
        We allow negative constants.

    Why:
        Some natural ARC formulas are like:

            width = -1 + 6*ring_count + 1*outside_blob_count

        If constants only start at 0, the learner can never find that.
    """
    formulas = []

    constants = list(range(-10, 21))
    weights = list(range(-5, 11))

    # --------------------------------------------------------
    # Constant-only formulas:
    #     value = constant
    # --------------------------------------------------------
    for constant in constants:
        if constant <= 0:
            continue

        formulas.append(
            {
                "constant": constant,
                "weights": {},
                "description": f"{constant}",
            }
        )

    # --------------------------------------------------------
    # One-feature formulas:
    #     value = constant + weight * feature
    # --------------------------------------------------------
    for feature_name in feature_names:
        if feature_name == "constant":
            continue

        for constant in constants:
            for weight in weights:
                if weight == 0:
                    continue

                formulas.append(
                    {
                        "constant": constant,
                        "weights": {
                            feature_name: weight,
                        },
                        "description": f"{constant} + {weight}*{feature_name}",
                    }
                )

    # --------------------------------------------------------
    # Two-feature formulas:
    #     value = constant + w1*f1 + w2*f2
    # --------------------------------------------------------
    useful_pairs = [
        ("ring_count", "blob_count"),
        ("ring_count", "inside_blob_count"),
        ("ring_count", "outside_blob_count"),
        ("blob_count", "outside_blob_count"),
        ("inside_blob_count", "outside_blob_count"),
        ("blob_count", "max_blobs_in_one_ring"),
        ("ring_count", "max_blobs_in_one_ring"),
        ("ring_plus_blob_count", "outside_blob_count"),
        ("ring_plus_inside_blob_count", "outside_blob_count"),
    ]

    for feature_a, feature_b in useful_pairs:
        if feature_a not in feature_names:
            continue

        if feature_b not in feature_names:
            continue

        for constant in constants:
            for weight_a in weights:
                for weight_b in weights:
                    # If this is a two-feature formula, both features
                    # must actually matter.
                    #
                    # Do NOT allow:
                    #     -1 + 6*ring_count + 0*outside_blob_count
                    #
                    # That pretends to be a two-feature formula while
                    # ignoring outside_blob_count.
                    if weight_a == 0 or weight_b == 0:
                        continue

                    formulas.append(
                        {
                            "constant": constant,
                            "weights": {
                                feature_a: weight_a,
                                feature_b: weight_b,
                            },
                            "description": (
                                f"{constant} + "
                                f"{weight_a}*{feature_a} + "
                                f"{weight_b}*{feature_b}"
                            ),
                        }
                    )

    return formulas


def formula_learning_priority(formula, dimension_index, examples):
    """
    Prefer formulas that use visually meaningful ARC features.
    """
    weights = formula.get("weights", {})

    score = 0

    uses_ring_count = "ring_count" in weights
    uses_blob_count = "blob_count" in weights
    uses_outside_blob_count = "outside_blob_count" in weights
    uses_inside_blob_count = "inside_blob_count" in weights
    uses_ring_plus_blob_count = "ring_plus_blob_count" in weights
    uses_ring_plus_inside_blob_count = "ring_plus_inside_blob_count" in weights

    if dimension_index == 0:
        if uses_ring_plus_blob_count:
            score -= 120

        if uses_ring_plus_inside_blob_count:
            score -= 90

        if uses_blob_count:
            score -= 60

        if uses_inside_blob_count:
            score -= 40

        if uses_ring_count:
            score -= 25

        if uses_outside_blob_count:
            score -= 10

    if dimension_index == 1:
        if uses_ring_count and uses_outside_blob_count:
            score -= 180

        if uses_ring_count:
            score -= 120

        if uses_ring_plus_blob_count:
            score -= 50

        if uses_ring_plus_inside_blob_count:
            score -= 50

        if uses_outside_blob_count:
            score -= 25

        if uses_outside_blob_count and not uses_ring_count:
            score += 80

    return score


def formula_complexity(formula):
    """
    Prefer simpler formulas when more than one formula fits.

    Simpler means:
        - fewer features
        - smaller constants
        - smaller weights

    This is not the only sorting rule.
    formula_learning_priority(...) gets used too.
    """
    weights = formula.get("weights", {})
    constant = formula.get("constant", 0)

    complexity = 0

    # Fewer moving parts is better.
    complexity += len(weights) * 100

    # Smaller constants are usually cleaner.
    complexity += abs(constant)

    # Smaller weights are usually cleaner.
    for weight in weights.values():
        complexity += abs(weight) * 5

    return complexity


def formula_learning_priority(formula, dimension_index, examples):
    """
    Prefer formulas that use visually meaningful ARC features.

    This is not task hardcoding.

    It is an inductive bias:
        - output height often grows with total visual complexity
        - output width often grows with number of rings / lanes
        - outside blobs may add lanes, but should not replace ring_count
    """
    weights = formula.get("weights", {})

    score = 0

    uses_ring_count = "ring_count" in weights
    uses_blob_count = "blob_count" in weights
    uses_outside_blob_count = "outside_blob_count" in weights
    uses_inside_blob_count = "inside_blob_count" in weights
    uses_ring_plus_blob_count = "ring_plus_blob_count" in weights
    uses_ring_plus_inside_blob_count = "ring_plus_inside_blob_count" in weights

    # --------------------------------------------------------
    # Height bias.
    #
    # For these ring/blob tasks, height often follows visual
    # complexity: rings + blobs.
    # --------------------------------------------------------
    if dimension_index == 0:
        if uses_ring_plus_blob_count:
            score -= 120

        if uses_ring_plus_inside_blob_count:
            score -= 90

        if uses_blob_count:
            score -= 60

        if uses_inside_blob_count:
            score -= 40

        if uses_ring_count:
            score -= 25

        if uses_outside_blob_count:
            score -= 10

    # --------------------------------------------------------
    # Width bias.
    #
    # Width should care strongly about ring_count because one-ring
    # and two-ring symbolic outputs have different widths.
    #
    # Outside blobs can add a side lane, but should not be the only
    # explanation when ring_count changes too.
    # --------------------------------------------------------
    if dimension_index == 1:
        # Strongly prefer the natural ring/blob width rule:
        #     width depends on ring_count,
        #     and outside blobs may add a side lane.
        if uses_ring_count and uses_outside_blob_count:
            score -= 180

        if uses_ring_count:
            score -= 120

        if uses_ring_plus_blob_count:
            score -= 50

        if uses_ring_plus_inside_blob_count:
            score -= 50

        if uses_outside_blob_count:
            score -= 25

        # Penalize formulas that use outside_blob_count alone
        # while ignoring ring_count.
        if uses_outside_blob_count and not uses_ring_count:
            score += 80

    return score


def formula_sort_key(formula, dimension_index, examples):
    """
    Sort formulas by:
        1. meaningful visual priority
        2. formula simplicity
    """
    return (
        formula_learning_priority(formula, dimension_index, examples),
        formula_complexity(formula),
    )


def learn_one_dimension_formula(records, dimension_index):
    """
    Learn one output dimension.

    dimension_index:
        0 = height
        1 = width
    """
    examples = []

    for record in records:
        signature = record.get("signature")
        output_shape = record.get("output_shape")

        if signature is None or output_shape is None:
            continue

        features = get_signature_feature_values(signature)
        target_value = output_shape[dimension_index]

        examples.append(
            {
                "features": features,
                "target": target_value,
                "pair_index": record.get("pair_index"),
            }
        )

    if not examples:
        return None

    feature_names = list(examples[0]["features"].keys())
    candidates = generate_candidate_formulas(feature_names)

    fitting_formulas = []

    for formula in candidates:
        fits_all = True

        for example in examples:
            predicted_value = evaluate_linear_formula(
                formula,
                example["features"],
            )

            if predicted_value != example["target"]:
                fits_all = False
                break

        if fits_all:
            fitting_formulas.append(formula)

    if not fitting_formulas:
        return None

    fitting_formulas.sort(
        key=lambda formula: formula_sort_key(
            formula,
            dimension_index,
            examples,
        )
    )

    return fitting_formulas[0]


def learn_symbolic_output_shape_rule(records):
    """
    Learn output height and width formulas from visible train records.

    This replaces hardcoded shape logic.
    """
    if not records:
        return None

    height_formula = learn_one_dimension_formula(
        records,
        dimension_index=0,
    )

    width_formula = learn_one_dimension_formula(
        records,
        dimension_index=1,
    )

    if height_formula is None or width_formula is None:
        return None

    return {
        "type": "learned_symbolic_output_shape_rule",
        "height_formula": height_formula,
        "width_formula": width_formula,
    }


def predict_shape_with_learned_symbolic_rule(shape_rule, signature):
    """
    Apply a learned symbolic shape rule to a new scene signature.
    """
    if shape_rule is None:
        return None

    features = get_signature_feature_values(signature)

    height = evaluate_linear_formula(
        shape_rule["height_formula"],
        features,
    )

    width = evaluate_linear_formula(
        shape_rule["width_formula"],
        features,
    )

    if height <= 0 or width <= 0:
        return None

    return height, width


def print_learned_shape_rule(shape_rule):
    if shape_rule is None:
        print("Learned shape rule: None")
        return

    height_formula = shape_rule.get("height_formula", {})
    width_formula = shape_rule.get("width_formula", {})

    print("Learned shape rule:")
    print(f"  height = {height_formula.get('description')}")
    print(f"  width  = {width_formula.get('description')}")


# ============================================================
# Prediction grid shaping
# ============================================================

def resize_grid_to_shape(grid, target_shape, fill_color):
    """
    Resize a grid by cropping or padding.

    This does not fully solve the drawing rule.
    It only ensures the output size comes from the learned shape rule
    instead of blindly staying on the selected template size.
    """
    if grid is None:
        return None

    if target_shape is None:
        return grid

    target_h, target_w = target_shape

    if target_h <= 0 or target_w <= 0:
        return grid

    new_grid = []

    old_h, old_w = grid_shape(grid)

    for r in range(target_h):
        new_row = []

        for c in range(target_w):
            if r < old_h and c < old_w:
                new_row.append(grid[r][c])
            else:
                new_row.append(fill_color)

        new_grid.append(new_row)

    return new_grid


# ============================================================
# Learned piece selection
# ============================================================

def learn_color_rule(records):
    """
    Simple color rule:
        output fill/foreground are record-specific,
        but prediction recolors selected templates using input colors.
    """
    return {
        "type": "input_to_output_recolor",
        "record_count": len(records),
    }


def find_exact_signature_record(rules, test_signature):
    for record in rules.get("records", []):
        train_signature = record.get("signature")

        if signatures_match(test_signature, train_signature):
            return record

    return None


def find_two_ring_compact_base(records):
    for record in records:
        sig = record.get("signature", {})

        if (
            sig.get("ring_count") == 2
            and sig.get("outside_blob_count") == 0
            and sig.get("ring_blob_counts") == (1, 1)
        ):
            return record

    return None


def find_two_ring_outside_lane_example(records):
    for record in records:
        sig = record.get("signature", {})

        if sig.get("outside_blob_count", 0) > 0:
            return record

    return None


def find_repeated_blob_example(records):
    for record in records:
        sig = record.get("signature", {})

        repeated = repeated_blob_ring_indexes(sig)

        if repeated:
            return record

    return None


def find_tall_two_ring_template(records):
    best = None
    best_height = -1

    for record in records:
        sig = record.get("signature", {})
        shape = record.get("output_shape", (0, 0))

        if sig.get("ring_count") != 2:
            continue

        height = shape[0]

        if height > best_height:
            best_height = height
            best = record

    return best


def record_similarity_score(record, test_signature):
    """
    Choose a reusable output template.

    This is still a template-choice step, but the final output size is
    corrected by the learned shape rule.
    """
    sig = record.get("signature", {})

    score = 0

    if sig.get("ring_count") == test_signature.get("ring_count"):
        score += 40
    else:
        score -= 20 * abs(sig.get("ring_count", 0) - test_signature.get("ring_count", 0))

    if sig.get("blob_count") == test_signature.get("blob_count"):
        score += 30
    else:
        score -= 10 * abs(sig.get("blob_count", 0) - test_signature.get("blob_count", 0))

    if sig.get("outside_blob_count") == test_signature.get("outside_blob_count"):
        score += 30
    else:
        score -= 15 * abs(
            sig.get("outside_blob_count", 0)
            - test_signature.get("outside_blob_count", 0)
        )

    if tuple(sig.get("ring_blob_counts", [])) == tuple(test_signature.get("ring_blob_counts", [])):
        score += 40

    if tuple(sig.get("ring_layouts", [])) == tuple(test_signature.get("ring_layouts", [])):
        score += 20

    return score


def choose_base_record_for_test(rules, test_signature):
    """
    Choose a base output template.

    Order:
        1. exact symbolic match, if available
        2. best scene similarity
        3. fallback first record

    The learned shape rule runs after this, so this base does not decide
    final output shape by itself.
    """
    exact_record = find_exact_signature_record(
        rules,
        test_signature,
    )

    if exact_record is not None:
        return exact_record, "exact_symbolic_scene_match"

    records = rules.get("records", [])

    if not records:
        return None, "no_records"

    best_record = None
    best_score = None

    for record in records:
        score = record_similarity_score(record, test_signature)

        if best_score is None or score > best_score:
            best_score = score
            best_record = record

    if best_record is not None:
        return best_record, f"best_scene_similarity_{best_score}"

    return records[0], "fallback_first_record"


# ============================================================
# Optional output composition helpers
# ============================================================

def build_outside_lane_from_all_train(base_grid, fill_color, foreground_color, outside_blob_count):
    """
    Compose a simple outside lane onto a base grid.

    This is still experimental.
    """
    if not base_grid:
        return base_grid

    h = len(base_grid)
    w = len(base_grid[0])

    core_width = min(w, 9)
    lane_width = 3

    new_grid = []

    for row in base_grid:
        core = row[:core_width]

        while len(core) < core_width:
            core.append(fill_color)

        new_grid.append(core + [fill_color] * lane_width)

    separator_col = core_width - 1

    for r in range(h):
        new_grid[r][separator_col] = foreground_color

    marker_col = core_width + 1

    if outside_blob_count <= 1:
        marker_rows = [h // 2]
    else:
        center = h // 2
        start = center - (outside_blob_count - 1)

        marker_rows = [
            start + 2 * index
            for index in range(outside_blob_count)
        ]

    for marker_row in marker_rows:
        if 0 <= marker_row < h and 0 <= marker_col < len(new_grid[0]):
            new_grid[marker_row][marker_col] = foreground_color

    return new_grid


def apply_repeated_blob_marker_from_all_train(grid, foreground_color, signature):
    """
    Experimental marker addition for repeated blobs.

    Kept simple for now. This will be improved after shape learning.
    """
    if grid is None:
        return None

    result = copy_grid(grid)

    repeated = repeated_blob_ring_indexes(signature)

    if not repeated:
        return result

    h, w = grid_shape(result)

    # Put a center marker if there is enough room.
    center_r = h // 2
    center_c = w // 2

    if 0 <= center_r < h and 0 <= center_c < w:
        result[center_r][center_c] = foreground_color

    return result


# ============================================================
# Main learner
# ============================================================

def learn_ring_blob_rule_synthesizer(train_pairs):
    """
    Learn from all visible train pairs.

    This builds:
        - records
        - learned shape formula
        - reusable template pieces
    """
    records = []

    for index, pair in enumerate(train_pairs):
        try:
            record = output_record_for_pair(pair, index)
            records.append(record)
        except Exception:
            continue

    shape_rule = learn_symbolic_output_shape_rule(records)

    rules = {
        "family": "ring_blob_rule_synthesizer",
        "type": "ring_blob_scene_to_output_template",
        "records": records,
        "record_count": len(records),
        "color_rule": learn_color_rule(records),
        "learned_shape_rule": shape_rule,

        # Reusable learned pieces.
        "two_ring_compact_base": find_two_ring_compact_base(records),
        "outside_lane_example": find_two_ring_outside_lane_example(records),
        "repeated_blob_example": find_repeated_blob_example(records),
        "tall_two_ring_template": find_tall_two_ring_template(records),
    }

    return rules


# ============================================================
# Main prediction
# ============================================================

def predict_with_ring_blob_rule_synthesizer(rules, input_grid):
    """
    Predict output for a new input grid.

    This does NOT receive expected output.

    Steps:
        1. read ring/blob scene
        2. choose a reusable base template
        3. recolor it
        4. apply learned symbolic shape rule
    """
    if rules is None:
        return {
            "prediction": None,
            "reason": "rules_none",
        }

    records = rules.get("records", [])

    if not records:
        return {
            "prediction": None,
            "reason": "no_records",
        }

    test_scene = learn_ring_blob_scene(input_grid)
    test_signature = scene_signature(test_scene)

    base_record, base_reason = choose_base_record_for_test(
        rules,
        test_signature,
    )

    if base_record is None:
        return {
            "prediction": None,
            "reason": "no_base_record",
            "test_signature": test_signature,
        }

    base_grid = copy_grid(base_record.get("output_grid"))

    old_fill = base_record.get("output_fill_color")
    old_foreground = base_record.get("output_foreground_color")

    new_fill = get_input_background_color(input_grid)
    new_foreground = get_input_foreground_color(input_grid)

    prediction = recolor_grid(
        base_grid,
        old_fill,
        old_foreground,
        new_fill,
        new_foreground,
    )

    # --------------------------------------------------------
    # Optional outside-lane composition.
    #
    # This is template help, not shape learning.
    # Shape is still corrected below by learned formula.
    # --------------------------------------------------------
    outside_blob_count = test_signature.get("outside_blob_count", 0)

    if (
        outside_blob_count > 0
        and base_record.get("signature", {}).get("outside_blob_count", 0) == 0
        and test_signature.get("ring_count", 0) >= 2
    ):
        prediction = build_outside_lane_from_all_train(
            prediction,
            new_fill,
            new_foreground,
            outside_blob_count,
        )

    # --------------------------------------------------------
    # Optional repeated-blob marker.
    # --------------------------------------------------------
    prediction = apply_repeated_blob_marker_from_all_train(
        prediction,
        new_foreground,
        test_signature,
    )

    # --------------------------------------------------------
    # Learned symbolic output shape correction.
    #
    # This is the important anti-hardcoding step.
    # The shape comes from formulas learned from visible train records.
    # --------------------------------------------------------
    symbolic_shape = predict_shape_with_learned_symbolic_rule(
        rules.get("learned_shape_rule"),
        test_signature,
    )

    if symbolic_shape is not None:
        fill_color = new_fill

        if fill_color is None:
            fill_color = get_output_fill_color_from_records(rules)

        if fill_color is None:
            fill_color = 0

        prediction = resize_grid_to_shape(
            prediction,
            symbolic_shape,
            fill_color,
        )

    return {
        "prediction": prediction,
        "test_scene": test_scene,
        "test_signature": test_signature,
        "base_record": base_record,
        "base_reason": base_reason,
        "learned_shape_rule": rules.get("learned_shape_rule"),
        "symbolic_shape": symbolic_shape,
    }


# ============================================================
# Debug / summary helpers
# ============================================================

def print_rule_synthesizer_summary(rules):
    if rules is None:
        print("RING/BLOB RULE SYNTHESIZER: None")
        return

    print()
    print("RING/BLOB RULE SYNTHESIZER — ALL-TRAIN SUMMARY")
    print("=" * 60)

    records = rules.get("records", [])

    for record in records:
        pair_index = record.get("pair_index")
        signature = record.get("signature")
        output_shape = record.get("output_shape")

        print()
        print(f"TRAIN PAIR {pair_index}")
        print("-" * 60)
        print(f"scene        : {signature_to_text(signature)}")
        print(f"output shape : {output_shape}")
        print(
            "colors       : "
            f"fill={record.get('output_fill_color')}, "
            f"foreground={record.get('output_foreground_color')}"
        )
        print(f"right lane   : {record.get('right_fill_lane_width')}")

    print()
    print("LEARNED SHAPE RULE")
    print("-" * 60)
    print_learned_shape_rule(rules.get("learned_shape_rule"))

    print()
    print("LEARNED PIECES")
    print("-" * 60)

    for key in [
        "two_ring_compact_base",
        "outside_lane_example",
        "repeated_blob_example",
        "tall_two_ring_template",
    ]:
        record = rules.get(key)

        if record is None:
            print(f"{key:28}: None")
        else:
            print(f"{key:28}: train pair {record.get('pair_index')}")


def print_rule_synthesizer_prediction_summary(result):
    if result is None:
        print("RING/BLOB PREDICTION: None")
        return

    prediction = result.get("prediction")
    signature = result.get("test_signature")
    base_record = result.get("base_record")

    print()
    print("RING/BLOB RULE SYNTHESIZER — PREDICTION SUMMARY")
    print("=" * 60)
    print(f"test scene       : {signature_to_text(signature)}")
    print(f"base reason      : {result.get('base_reason')}")

    if base_record is None:
        print("base train pair  : None")
    else:
        print(f"base train pair  : {base_record.get('pair_index')}")

    print(f"symbolic shape   : {result.get('symbolic_shape')}")

    print_learned_shape_rule(result.get("learned_shape_rule"))

    h, w = grid_shape(prediction)
    print(f"prediction shape : {h}x{w}")


# ============================================================
# Router compatibility helpers
# ============================================================

def score_prediction(predicted, expected):
    """
    Local score helper for this strategy wrapper.
    """
    if predicted is None or expected is None:
        return 0

    ph, pw = grid_shape(predicted)
    eh, ew = grid_shape(expected)

    score = 0

    for r in range(min(ph, eh)):
        for c in range(min(pw, ew)):
            if predicted[r][c] == expected[r][c]:
                score += 1

    if predicted == expected:
        score += 1_000_000

    return score


def solve_with_ring_blob_rule_synthesizer(input_grid, output_grid):
    """
    Pair-level compatibility wrapper.

    This is not the main way we want to use the synthesizer.
    The main way is:
        learn_ring_blob_rule_synthesizer(train_pairs)
        predict_with_ring_blob_rule_synthesizer(rules, test_input)

    This wrapper exists only so older router imports do not break.
    """
    train_pairs = [
        {
            "input": input_grid,
            "output": output_grid,
        }
    ]

    rules = learn_ring_blob_rule_synthesizer(train_pairs)
    prediction_result = predict_with_ring_blob_rule_synthesizer(
        rules,
        input_grid,
    )

    predicted = prediction_result.get("prediction")

    return {
        "strategy": "ring_blob_rule_synthesizer",
        "predicted": predicted,
        "prediction": predicted,
        "score": score_prediction(predicted, output_grid),
        "exact": predicted == output_grid,
        "rule": rules,
    }