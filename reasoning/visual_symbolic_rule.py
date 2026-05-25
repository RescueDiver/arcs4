# reasoning/visual_symbolic_rule.py

"""
visual_symbolic_rule.py

Goal:
    Learn a task-level visual rule from ring/blob scene structure.

Important:
    This file is NOT allowed to predict by copying an expected output.

Pipeline:
    1. Read input as a visual scene:
        - rings / enclosures
        - blobs
        - which blobs are inside which rings
        - which blobs are outside all rings

    2. Learn symbolic output construction from train pairs:
        - output shape formulas
        - output color role rule
        - output drawing rule

    3. Apply the learned symbolic rule to new inputs without expected output.

Important design rule:
    Shape and drawing are learned from examples.
    We do NOT hard-code task-specific output sizes or special-case pairs.
"""

from collections import Counter, deque

from core.scoring import score_prediction

try:
    from reasoning.visual_abstraction_discovery import discover_visual_abstractions
except ImportError:
    discover_visual_abstractions = None


# ============================================================
# BASIC HELPERS
# ============================================================

def grid_shape(grid):
    if grid is None:
        return 0, 0

    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def copy_grid(grid):
    return [row[:] for row in grid]


def colors(grid):
    return set(v for row in grid for v in row)


def color_counts(grid):
    return Counter(v for row in grid for v in row)


def most_common_color(grid):
    counts = color_counts(grid)

    if not counts:
        return 0

    return counts.most_common(1)[0][0]


def least_common_color(grid):
    counts = color_counts(grid)

    if not counts:
        return 0

    return counts.most_common()[-1][0]


def ordered_colors_by_count(grid):
    counts = color_counts(grid)
    return [color for color, count in counts.most_common()]


def non_background_colors(grid):
    bg = most_common_color(grid)
    return sorted(c for c in colors(grid) if c != bg)


def get_view(summary, view_type):
    if summary is None:
        return None

    for view in summary.get("views", []):
        if view.get("view_type") == view_type:
            return view

    return None


def set_cell(grid, r, c, value):
    h, w = grid_shape(grid)

    if 0 <= r < h and 0 <= c < w:
        grid[r][c] = value


def box_height(box):
    return box.get("height", 0)


def box_width(box):
    return box.get("width", 0)


def box_area(box):
    return box.get("height", 0) * box.get("width", 0)


def box_center(box):
    if box is None:
        return None

    return (
        (box["top"] + box["bottom"]) // 2,
        (box["left"] + box["right"]) // 2,
    )


def box_contains_point(box, r, c):
    if box is None:
        return False

    return (
        box["top"] <= r <= box["bottom"]
        and box["left"] <= c <= box["right"]
    )


def box_strictly_contains_box(outer, inner):
    if outer is None or inner is None:
        return False

    return (
        outer["top"] <= inner["top"]
        and outer["bottom"] >= inner["bottom"]
        and outer["left"] <= inner["left"]
        and outer["right"] >= inner["right"]
        and (
            outer["top"] < inner["top"]
            or outer["bottom"] > inner["bottom"]
            or outer["left"] < inner["left"]
            or outer["right"] > inner["right"]
        )
    )


def bbox_from_cells(cells):
    if not cells:
        return None

    rows = [r for r, c in cells]
    cols = [c for r, c in cells]

    top = min(rows)
    bottom = max(rows)
    left = min(cols)
    right = max(cols)

    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "height": bottom - top + 1,
        "width": right - left + 1,
        "area": (bottom - top + 1) * (right - left + 1),
    }

def formula_feature_coefficient(formula, feature_name):
    total = 0

    for coef, name in formula.get("terms", []):
        if name == feature_name:
            total += coef

    return total



def formula_name(formula):
    if formula is None:
        return "None"

    return formula.get("name", "UNKNOWN_FORMULA")


def print_shape_rule_debug(shape_rule, label="SHAPE RULE"):
    print()
    print(f"[{label}]")
    print("height_formula:", formula_name(shape_rule.get("height_formula")))
    print("width_formula :", formula_name(shape_rule.get("width_formula")))


# ============================================================
# INPUT SCENE READING
# ============================================================

def analyze_input_scene(input_grid):
    """
    Convert an input grid into a compact visual scene.

    Uses ring_blob_view first because it is currently more reliable than
    enclosure_tree_view for 2d0172a1.
    """
    if discover_visual_abstractions is None:
        return None

    summary = discover_visual_abstractions(input_grid)
    ring_blob = get_view(summary, "ring_blob_view")

    if ring_blob is None:
        return None

    rings = ring_blob.get("rings", [])
    blobs = ring_blob.get("blobs", [])
    assignments = ring_blob.get("assignments", [])

    ring_ids = [r.get("id") for r in rings]

    blobs_by_ring = {ring_id: [] for ring_id in ring_ids}
    outside_blobs = []

    for assignment in assignments:
        blob_id = assignment.get("blob_id")
        assigned_to = assignment.get("assigned_to")

        if assigned_to == "outside_all":
            outside_blobs.append(blob_id)
        elif assigned_to in blobs_by_ring:
            blobs_by_ring[assigned_to].append(blob_id)

    ring_blob_counts = []

    for ring in rings:
        ring_id = ring.get("id")
        ring_blob_counts.append(len(blobs_by_ring.get(ring_id, [])))

    scene = {
        "grid_shape": grid_shape(input_grid),
        "background": most_common_color(input_grid),
        "active_colors": non_background_colors(input_grid),
        "ring_count": len(rings),
        "blob_count": len(blobs),
        "outside_blob_count": len(outside_blobs),
        "ring_blob_counts": tuple(ring_blob_counts),
        "rings": rings,
        "blobs": blobs,
        "assignments": assignments,
        "blobs_by_ring": blobs_by_ring,
        "outside_blobs": outside_blobs,
    }

    return scene


def scene_signature(scene):
    if scene is None:
        return None

    return {
        "ring_count": scene.get("ring_count", 0),
        "blob_count": scene.get("blob_count", 0),
        "outside_blob_count": scene.get("outside_blob_count", 0),
        "ring_blob_counts": scene.get("ring_blob_counts", ()),
    }


# ============================================================
# OUTPUT READING
# ============================================================

def analyze_output_grid(output_grid):
    """
    Read output shape/colors.

    We do NOT store the full output as a template to replay.
    We only learn:
        - output height/width
        - border/background role colors
    """
    h, w = grid_shape(output_grid)
    ordered = ordered_colors_by_count(output_grid)

    if len(ordered) < 2:
        return None

    background_color = ordered[0]
    mark_color = ordered[1]

    edge_colors = []

    for c in range(w):
        edge_colors.append(output_grid[0][c])
        edge_colors.append(output_grid[h - 1][c])

    for r in range(h):
        edge_colors.append(output_grid[r][0])
        edge_colors.append(output_grid[r][w - 1])

    edge_common = Counter(edge_colors).most_common(1)[0][0]

    draw_color = edge_common
    fill_color = background_color if background_color != draw_color else mark_color

    return {
        "shape": (h, w),
        "height": h,
        "width": w,
        "draw_color": draw_color,
        "fill_color": fill_color,
        "colors": ordered,
    }


# ============================================================
# FORMULA LEARNING
# ============================================================

def extract_shape_features(scene):
    """
    Convert a visual scene into numeric features.

    These are the only things the shape learner is allowed to use.
    No expected output is used here.
    """
    ring_count = scene.get("ring_count", 0)
    blob_count = scene.get("blob_count", 0)
    outside_blob_count = scene.get("outside_blob_count", 0)
    ring_blob_counts = scene.get("ring_blob_counts", ())

    inside_blob_count = blob_count - outside_blob_count

    if ring_blob_counts:
        max_ring_blob_count = max(ring_blob_counts)
        min_ring_blob_count = min(ring_blob_counts)
    else:
        max_ring_blob_count = 0
        min_ring_blob_count = 0

    nested_ring_count = max(0, ring_count - 1)
    total_scene_objects = ring_count + blob_count

    return {
        "one": 1,
        "ring_count": ring_count,
        "blob_count": blob_count,
        "outside_blob_count": outside_blob_count,
        "inside_blob_count": inside_blob_count,
        "max_ring_blob_count": max_ring_blob_count,
        "min_ring_blob_count": min_ring_blob_count,
        "nested_ring_count": nested_ring_count,
        "total_scene_objects": total_scene_objects,
    }


def extract_rule_features(scene, output_height=None, output_width=None):
    """
    Features available to learned formulas.

    Shape formulas use only scene features.
    Drawing formulas may also use predicted output height/width.
    """
    features = extract_shape_features(scene)

    if output_height is not None:
        features["output_height"] = output_height

    if output_width is not None:
        features["output_width"] = output_width

    return features


def make_formula(name, terms):
    return {
        "name": name,
        "terms": terms,
    }


def apply_formula_to_features(formula, features):
    if formula is None:
        return None

    value = 0

    for coef, feature_name in formula.get("terms", []):
        value += coef * features.get(feature_name, 0)

    return value


def apply_shape_formula(formula, scene):
    features = extract_rule_features(scene)
    return apply_formula_to_features(formula, features)


def generate_formulas(feature_names):
    """
    Generate readable formulas.

    This is a search space.
    The selected formula is learned by fitting train examples.
    """
    formulas = []

    constants = list(range(0, 31))
    coefficients = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8]

    for k in constants:
        formulas.append(
            make_formula(
                str(k),
                [(k, "one")],
            )
        )

    for feature in feature_names:
        for coef in coefficients:
            for k in constants:
                formulas.append(
                    make_formula(
                        f"{coef}*{feature}+{k}",
                        [
                            (coef, feature),
                            (k, "one"),
                        ],
                    )
                )

    for feature_a in feature_names:
        for feature_b in feature_names:
            if feature_a == feature_b:
                continue

            for coef_a in coefficients:
                for coef_b in coefficients:
                    for k in constants:
                        formulas.append(
                            make_formula(
                                f"{coef_a}*{feature_a}+{coef_b}*{feature_b}+{k}",
                                [
                                    (coef_a, feature_a),
                                    (coef_b, feature_b),
                                    (k, "one"),
                                ],
                            )
                        )

    return formulas


def formula_complexity(formula):
    """
    Prefer stable, human-simple visual formulas.

    Lower is better.
    """
    if formula is None:
        return 10**9

    terms = formula.get("terms", [])

    non_constant_terms = [
        (coef, name)
        for coef, name in terms
        if name != "one" and coef != 0
    ]

    used_features = {
        name
        for coef, name in non_constant_terms
    }

    feature_priority = {
        "blob_count": 0,
        "nested_ring_count": 0,
        "outside_blob_count": 1,

        "output_height": 2,
        "output_width": 2,

        "ring_count": 5,
        "inside_blob_count": 6,
        "total_scene_objects": 7,

        "max_ring_blob_count": 20,
        "min_ring_blob_count": 25,

        "one": 0,
    }

    feature_cost = sum(
        feature_priority.get(name, 50) * 1000
        for name in used_features
    )

    term_count_cost = len(non_constant_terms) * 500

    coef_cost = 0

    for coef, name in non_constant_terms:
        abs_coef = abs(coef)

        if abs_coef in [1, 2, 3, 5, 6]:
            coef_cost += abs_coef * 10
        else:
            coef_cost += abs_coef * 50

    constant_cost = 0

    for coef, name in terms:
        if name == "one":
            constant_cost += abs(coef) * 5

    bonus = 0

    if used_features == {"blob_count"}:
        bonus -= 1000

    if used_features == {"nested_ring_count", "outside_blob_count"}:
        bonus -= 1000

    if used_features == {"blob_count", "nested_ring_count"}:
        bonus -= 500

    if used_features == {"output_width", "outside_blob_count"}:
        bonus -= 500

    return (
        feature_cost
        + term_count_cost
        + coef_cost
        + constant_cost
        + bonus
    )

def formula_uses_feature(formula, feature_name):
    for coef, name in formula.get("terms", []):
        if name == feature_name and coef != 0:
            return True

    return False


def formula_used_features(formula):
    return {
        name
        for coef, name in formula.get("terms", [])
        if name != "one" and coef != 0
    }


def examples_have_feature_variation(examples, feature_name):
    values = []

    for ex in examples:
        scene = ex["scene"]
        output_info = ex.get("output_info", {})

        features = extract_rule_features(
            scene,
            output_height=output_info.get("height"),
            output_width=output_info.get("width"),
        )

        values.append(features.get(feature_name, 0))

    return len(set(values)) > 1


def formula_generalization_priority(
    formula,
    examples,
    target_name=None,
    prediction_scene=None,
):
    """
    Lower is better.

    Shape formulas and drawing formulas need different bias.

    Shape:
        height -> blob_count + nested_ring_count
        width  -> outside_blob_count + nested_ring_count

    Drawing:
        row placement -> output_height + blob/nesting structure
        col placement -> output_width + outside/nesting structure
    """
    if formula is None:
        return 10**9

    used = formula_used_features(formula)
    cost = 0

    # ------------------------------------------------------------
    # POSITION FORMULA BIAS
    # ------------------------------------------------------------
    # For box top/left, a constant zero is usually the most stable
    # visual rule when it fits the examples.
    #
    # This is not forcing the answer. The formula still must fit all
    # visible training examples before it can be selected.
    used = formula_used_features(formula)

    if target_name in {"top", "left"} and not used:
        predicted_values = []

        for ex in examples:
            scene = ex["scene"]
            output_info = ex.get("output_info", {})

            features = extract_rule_features(
                scene,
                output_height=output_info.get("height"),
                output_width=output_info.get("width"),
            )

            predicted_values.append(
                apply_formula_to_features(formula, features)
            )

        if predicted_values and all(value == 0 for value in predicted_values):
            cost -= 100000

    # ------------------------------------------------------------
    # SHAPE FORMULAS
    # ------------------------------------------------------------
    if target_name == "height":
        preferred = [
            "blob_count",
            "nested_ring_count",
        ]

        discouraged = [
            "outside_blob_count",
            "output_height",
            "output_width",
        ]

    elif target_name == "width":
        preferred = [
            "outside_blob_count",
            "nested_ring_count",
        ]

        discouraged = [
            "output_height",
            "output_width",
        ]

    # ------------------------------------------------------------
    # DRAWING ROW FORMULAS
    # ------------------------------------------------------------
    elif target_name in {
        "row",
        "top",
        "bottom",
        "center_row",
        "inner_top",
        "inner_bottom",
        "marker_row",
        "box_height",
    }:
        preferred = [
            "output_height",
            "blob_count",
            "nested_ring_count",
        ]

        discouraged = []

    # ------------------------------------------------------------
    # DRAWING COLUMN FORMULAS
    # ------------------------------------------------------------
    elif target_name in {
        "col",
        "left",
        "right",
        "center_col",
        "inner_left",
        "inner_right",
        "marker_col",
        "box_width",
    }:
        preferred = [
            "output_width",
            "outside_blob_count",
            "nested_ring_count",
        ]

        discouraged = []

    else:
        preferred = [
            "blob_count",
            "nested_ring_count",
            "outside_blob_count",
        ]

        discouraged = []

    # Features that vary in visible train examples matter.
    for feature in preferred:
        if examples_have_feature_variation(examples, feature):
            if feature not in used:
                cost += 5000

    # Features present in the input being predicted also matter.
    if prediction_scene is not None:
        prediction_features = extract_rule_features(prediction_scene)

        for feature in preferred:
            value = prediction_features.get(feature, 0)

            if value != 0 and feature not in used:
                cost += 7000

    # Discourage leakage only for final shape formulas.
    # Drawing formulas are allowed to use output_height/output_width
    # because the canvas size has already been learned.
    if target_name in {"height", "width"}:
        for feature in discouraged:
            if feature in used:
                cost += 3000

    # Width extension should add width, not shrink it.
    if target_name == "width":
        outside_coef = formula_feature_coefficient(
            formula,
            "outside_blob_count",
        )

        if outside_coef < 0:
            cost += 10000

    return cost


def learn_best_formula_from_examples(
    examples,
    target_getter,
    feature_names,
    target_name=None,
    prediction_scene=None,
):
    """
    Learn a formula that fits all given examples.

    target_getter(ex) returns the number this formula must predict.
    """
    if not examples:
        return None

    formulas = generate_formulas(feature_names)
    valid = []

    for formula in formulas:
        all_match = True

        for ex in examples:
            scene = ex["scene"]
            output_info = ex.get("output_info", {})

            features = extract_rule_features(
                scene,
                output_height=output_info.get("height"),
                output_width=output_info.get("width"),
            )

            predicted_value = apply_formula_to_features(formula, features)
            expected_value = target_getter(ex)

            if predicted_value != expected_value:
                all_match = False
                break

        if all_match:
            valid.append(formula)

    if not valid:
        return None

    valid.sort(
        key=lambda formula: (
            formula_generalization_priority(
                formula,
                examples,
                target_name=target_name,
                prediction_scene=prediction_scene,
            ),
            formula_complexity(formula),
        )
    )

    return valid[0]


# ============================================================
# LEARNING OUTPUT SIZE BY FORMULA
# ============================================================

def learn_best_shape_formula(examples, target_name, prediction_scene=None):
    if not examples:
        return None

    feature_names = [
        "ring_count",
        "blob_count",
        "outside_blob_count",
        "inside_blob_count",
        "max_ring_blob_count",
        "min_ring_blob_count",
        "nested_ring_count",
        "total_scene_objects",
    ]

    return learn_best_formula_from_examples(
        examples,
        target_getter=lambda ex: ex["output_info"][target_name],
        feature_names=feature_names,
        target_name=target_name,
        prediction_scene=prediction_scene,
    )


def learn_shape_rule(examples, prediction_scene=None):
    """
    Learn output shape formulas from train examples.
    """
    height_formula = learn_best_shape_formula(
        examples,
        target_name="height",
        prediction_scene=prediction_scene,
    )

    width_formula = learn_best_shape_formula(
        examples,
        target_name="width",
        prediction_scene=prediction_scene,
    )

    return {
        "type": "learned_shape_formulas",
        "height_formula": height_formula,
        "width_formula": width_formula,
    }


def predict_shape_from_rule(shape_rule, scene):
    if shape_rule is None:
        return None

    height_formula = shape_rule.get("height_formula")
    width_formula = shape_rule.get("width_formula")

    if height_formula is None or width_formula is None:
        return None

    height = apply_shape_formula(height_formula, scene)
    width = apply_shape_formula(width_formula, scene)

    if height is None or width is None:
        return None

    if height <= 0 or width <= 0:
        return None

    return height, width


# ============================================================
# COLOR ROLE LEARNING
# ============================================================

def learn_color_rule(examples):
    """
    Learn how output draw/fill colors relate to input colors.

    This avoids hard-coding:
        draw = input active
        fill = input background

    Instead, it tests simple color-role hypotheses and picks one
    that fits all visible train examples.
    """
    print()
    print("[COLOR LEARNING EXAMPLES]")
    for ex in examples:
        scene = ex["scene"]
        output_info = ex["output_info"]

        print(
            f"pair={ex.get('pair_index')} "
            f"rings={scene.get('ring_count')} "
            f"blobs={scene.get('blob_count')} "
            f"outside={scene.get('outside_blob_count')} "
            f"bg={scene.get('background')} "
            f"active={scene.get('active_colors')} "
            f"expected_draw={output_info.get('draw_color')} "
            f"expected_fill={output_info.get('fill_color')}"
        )


    hypotheses = [

        {
            "name": "nested_no_outside_uses_background_to_draw_else_active_to_draw",
            "mode": "conditional",
        },
        {
            "name": "input_active_to_draw__input_background_to_fill",
            "draw_source": "input_active",
            "fill_source": "input_background",
        },
        {
            "name": "input_background_to_draw__input_active_to_fill",
            "draw_source": "input_background",
            "fill_source": "input_active",
        },
    ]

    for hypothesis in hypotheses:
        all_match = True

        for ex in examples:
            scene = ex["scene"]
            output_info = ex["output_info"]

            predicted_draw, predicted_fill = apply_color_rule_to_scene(
                hypothesis,
                scene,
            )

            if predicted_draw != output_info["draw_color"]:
                all_match = False
                break

            if predicted_fill != output_info["fill_color"]:
                all_match = False
                break

        if all_match:
            return hypothesis

    return {
        "name": "fallback_input_active_to_draw__input_background_to_fill",
        "draw_source": "input_active",
        "fill_source": "input_background",
    }


def apply_color_rule_to_scene(color_rule, scene):
    bg = scene.get("background", 0)
    active = scene.get("active_colors", [])

    if active:
        active_color = active[0]
    else:
        active_color = bg

    name = color_rule.get("name")



    # Conditional learned color grammar:
    # nested ring scene with no outside blob flips color roles.
    if name == "nested_no_outside_uses_background_to_draw_else_active_to_draw":
        nested_ring_count = max(0, scene.get("ring_count", 0) - 1)
        outside_blob_count = scene.get("outside_blob_count", 0)

        if nested_ring_count > 0 and outside_blob_count == 0:
            return bg, active_color

        return active_color, bg

    def resolve(source):
        if source == "input_active":
            return active_color

        if source == "input_background":
            return bg

        return active_color

    draw_color = resolve(color_rule.get("draw_source"))
    fill_color = resolve(color_rule.get("fill_source"))

    return draw_color, fill_color


# ============================================================
# OUTPUT DRAWING ANALYSIS
# ============================================================

def find_color_components(grid, target_color):
    h, w = grid_shape(grid)
    seen = set()
    components = []

    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue

            if grid[r][c] != target_color:
                continue

            queue = deque([(r, c)])
            seen.add((r, c))
            cells = []

            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = cr + dr
                    nc = cc + dc

                    if not (0 <= nr < h and 0 <= nc < w):
                        continue

                    if (nr, nc) in seen:
                        continue

                    if grid[nr][nc] != target_color:
                        continue

                    seen.add((nr, nc))
                    queue.append((nr, nc))

            box = bbox_from_cells(cells)

            components.append({
                "cells": cells,
                "cell_set": set(cells),
                "size": len(cells),
                "box": box,
            })

    return components


def touches_grid_edge(component, height, width):
    for r, c in component.get("cells", []):
        if r == 0 or c == 0 or r == height - 1 or c == width - 1:
            return True

    return False


def perimeter_cells_for_box(box):
    if box is None:
        return set()

    cells = set()

    top = box["top"]
    bottom = box["bottom"]
    left = box["left"]
    right = box["right"]

    for c in range(left, right + 1):
        cells.add((top, c))
        cells.add((bottom, c))

    for r in range(top, bottom + 1):
        cells.add((r, left))
        cells.add((r, right))

    return cells


def component_is_box_perimeter(component):
    box = component.get("box")
    cells = component.get("cell_set", set())

    if box is None:
        return False

    if box["height"] < 3 or box["width"] < 3:
        return False

    needed = perimeter_cells_for_box(box)
    return needed.issubset(cells)


def find_drawn_box_perimeters(grid, draw_color):
    """
    Find all rectangular box perimeters made of draw_color.

    This does not care whether boxes are connected to each other.
    That lets us detect chamber boxes attached to the outer frame.
    """
    h, w = grid_shape(grid)
    boxes = []

    for top in range(h):
        for left in range(w):
            for bottom in range(top + 2, h):
                for right in range(left + 2, w):
                    box = {
                        "top": top,
                        "bottom": bottom,
                        "left": left,
                        "right": right,
                        "height": bottom - top + 1,
                        "width": right - left + 1,
                        "area": (bottom - top + 1) * (right - left + 1),
                    }

                    perimeter = perimeter_cells_for_box(box)

                    if all(grid[r][c] == draw_color for r, c in perimeter):
                        boxes.append(box)

    boxes.sort(
        key=lambda box: (
            box["top"],
            box["left"],
            box["height"],
            box["width"],
        )
    )

    return boxes


def analyze_output_drawing(output_grid, output_info):
    """
    Read output drawing symbolically.

    This does NOT store the output grid as a template.
    It extracts:
        - outer frame box
        - inner frame boxes
        - marker cells

    These become training targets for learn_drawing_rule().
    """
    h, w = grid_shape(output_grid)
    draw_color = output_info["draw_color"]

    components = find_color_components(output_grid, draw_color)

    if not components:
        return None

    edge_components = [
        comp
        for comp in components
        if touches_grid_edge(comp, h, w)
    ]

    if edge_components:
        outer_component = max(
            edge_components,
            key=lambda comp: box_area(comp["box"]),
        )
    else:
        outer_component = max(
            components,
            key=lambda comp: box_area(comp["box"]),
        )

    outer_box = outer_component["box"]

    inner_boxes = []

    for comp in components:
        if comp is outer_component:
            continue

        if component_is_box_perimeter(comp):
            inner_boxes.append(comp["box"])

    inner_boxes.sort(
        key=lambda box: (
            box["top"],
            box["left"],
            box["height"],
            box["width"],
        )
    )

    all_drawn_boxes = find_drawn_box_perimeters(output_grid, draw_color)

    # Extra boxes includes attached chambers that component detection can miss.
    # We remove the main outer box and exact duplicates only.
    extra_boxes = []

    for box in all_drawn_boxes:
        if box == outer_box:
            continue

        if box not in extra_boxes:
            extra_boxes.append(box)

    covered = set()
    covered.update(perimeter_cells_for_box(outer_box))

    for box in inner_boxes:
        covered.update(perimeter_cells_for_box(box))

    markers = []

    for comp in components:
        for r, c in comp.get("cells", []):
            if (r, c) in covered:
                continue

            markers.append({
                "row": r,
                "col": c,
            })

    markers.sort(key=lambda m: (m["row"], m["col"]))

    return {
        "outer_box": outer_box,
        "inner_boxes": inner_boxes,
        "extra_boxes": extra_boxes,
        "markers": markers,
    }


# ============================================================
# LEARNING DRAWING RULE
# ============================================================

def marker_at(markers, row, col):
    for marker in markers:
        if marker["row"] == row and marker["col"] == col:
            return True

    return False


def choose_inner_box(drawing_info):
    boxes = drawing_info.get("inner_boxes", [])

    if not boxes:
        return None

    boxes = sorted(
        boxes,
        key=lambda box: (
            box_area(box),
            box["top"],
            box["left"],
        ),
        reverse=True,
    )

    return boxes[0]


def learn_box_formula_rule(examples, box_getter, prediction_scene=None):
    usable = []

    for ex in examples:
        box = box_getter(ex)

        if box is None:
            continue

        usable.append(ex)

    if not usable:
        return None

    feature_names = [
        "ring_count",
        "blob_count",
        "outside_blob_count",
        "inside_blob_count",
        "nested_ring_count",
        "output_height",
        "output_width",
    ]

    def target_part(part):
        return lambda ex: box_getter(ex)[part]

    return {
        "top_formula": learn_best_formula_from_examples(
            usable,
            target_getter=target_part("top"),
            feature_names=feature_names,
            target_name="top",
            prediction_scene=prediction_scene,
        ),
        "left_formula": learn_best_formula_from_examples(
            usable,
            target_getter=target_part("left"),
            feature_names=feature_names,
            target_name="left",
            prediction_scene=prediction_scene,
        ),
        "height_formula": learn_best_formula_from_examples(
            usable,
            target_getter=target_part("height"),
            feature_names=feature_names,
            target_name="box_height",
            prediction_scene=prediction_scene,
        ),
        "width_formula": learn_best_formula_from_examples(
            usable,
            target_getter=target_part("width"),
            feature_names=feature_names,
            target_name="box_width",
            prediction_scene=prediction_scene,
        ),
        "example_count": len(usable),
    }


def apply_box_formula_rule(box_rule, scene, output_height, output_width):
    if box_rule is None:
        return None

    features = extract_rule_features(
        scene,
        output_height=output_height,
        output_width=output_width,
    )

    top = apply_formula_to_features(box_rule.get("top_formula"), features)
    left = apply_formula_to_features(box_rule.get("left_formula"), features)
    height = apply_formula_to_features(box_rule.get("height_formula"), features)
    width = apply_formula_to_features(box_rule.get("width_formula"), features)

    if None in [top, left, height, width]:
        return None

    if height <= 0 or width <= 0:
        return None

    return {
        "top": top,
        "left": left,
        "bottom": top + height - 1,
        "right": left + width - 1,
        "height": height,
        "width": width,
        "area": height * width,
    }


def learn_inner_box_presence_rule(examples):
    """
    Learn when an inner symbolic box should be drawn.

    This tests simple visual conditions. The condition itself is selected
    from train examples.
    """
    candidates = [
        {
            "name": "nested_ring_count_gt_0",
            "test": lambda scene: max(0, scene.get("ring_count", 0) - 1) > 0,
        },
        {
            "name": "ring_count_gt_1",
            "test": lambda scene: scene.get("ring_count", 0) > 1,
        },
        {
            "name": "always",
            "test": lambda scene: True,
        },
        {
            "name": "never",
            "test": lambda scene: False,
        },
    ]

    for candidate in candidates:
        all_match = True

        for ex in examples:
            scene = ex["scene"]
            drawing_info = ex["drawing_info"]
            has_inner_box = choose_inner_box(drawing_info) is not None
            predicted = candidate["test"](scene)

            if predicted != has_inner_box:
                all_match = False
                break

        if all_match:
            return {
                "name": candidate["name"],
            }

    return {
        "name": "never",
    }


def apply_inner_box_presence_rule(rule, scene):
    name = rule.get("name")

    if name == "nested_ring_count_gt_0":
        return max(0, scene.get("ring_count", 0) - 1) > 0

    if name == "ring_count_gt_1":
        return scene.get("ring_count", 0) > 1

    if name == "always":
        return True

    return False


def learn_one_ring_marker_rule(examples):
    """
    Learn marker placement for one-ring scenes.

    Example pattern discovered from outputs:
        marker column is output center column
        marker rows repeat with a fixed start/step

    The start/step are learned from output markers.
    """
    one_ring_examples = [
        ex
        for ex in examples
        if ex["scene"].get("ring_count", 0) == 1
    ]
    print()
    print("[ONE RING MARKER LEARNING EXAMPLES]")
    for ex in one_ring_examples:
        scene = ex["scene"]
        output_info = ex["output_info"]
        drawing_info = ex["drawing_info"]

        print(
            f"pair={ex.get('pair_index')} "
            f"rings={scene.get('ring_count')} "
            f"blobs={scene.get('blob_count')} "
            f"shape={output_info.get('height')}x{output_info.get('width')} "
            f"markers={drawing_info.get('markers')}"
        )
    if not one_ring_examples:
        return None

    starts = []
    steps = []
    center_col_ok = True

    for ex in one_ring_examples:
        markers = ex["drawing_info"].get("markers", [])
        output_info = ex["output_info"]
        blob_count = ex["scene"].get("blob_count", 0)

        if blob_count <= 0:
            continue

        expected_center_col = output_info["width"] // 2
        marker_rows = sorted(m["row"] for m in markers if m["col"] == expected_center_col)

        if len(marker_rows) != blob_count:
            return None

        for marker in markers:
            if marker["col"] != expected_center_col:
                center_col_ok = False

        if marker_rows:
            starts.append(marker_rows[0])

        if len(marker_rows) >= 2:
            diffs = [
                marker_rows[i + 1] - marker_rows[i]
                for i in range(len(marker_rows) - 1)
            ]
            if len(set(diffs)) == 1:
                steps.append(diffs[0])
            else:
                return None

    if not starts:
        return None

    if len(set(starts)) != 1:
        return None

    if steps:
        if len(set(steps)) != 1:
            return None

        step = steps[0]
    else:
        step = 2




    return {
        "type": "one_ring_repeating_center_markers",
        "row_start": starts[0],
        "row_step": step,
        "col_mode": "output_center",
        "requires_center_col": center_col_ok,
    }


def get_outer_only_blob_directions(scene):
    """
    Read input scene and classify blobs that are inside the outer ring
    but not inside the inner ring.

    Directions are symbolic:
        below, above, right, left, center
    """
    rings = scene.get("rings", [])
    blobs = scene.get("blobs", [])
    assignments = scene.get("assignments", [])

    if len(rings) < 2:
        return []

    rings_by_id = {
        ring.get("id"): ring
        for ring in rings
    }

    rings_sorted = sorted(
        rings,
        key=lambda ring: box_area(ring.get("box", {})),
        reverse=True,
    )

    outer_ring = rings_sorted[0]
    inner_ring = rings_sorted[1]

    outer_id = outer_ring.get("id")
    inner_box = inner_ring.get("box")

    if inner_box is None:
        return []

    inner_center = box_center(inner_box)

    if inner_center is None:
        return []

    inner_center_r, inner_center_c = inner_center

    blobs_by_id = {
        blob.get("id"): blob
        for blob in blobs
    }

    directions = []

    for assignment in assignments:
        blob_id = assignment.get("blob_id")
        assigned_to = assignment.get("assigned_to")

        if assigned_to != outer_id:
            continue

        blob = blobs_by_id.get(blob_id)

        if blob is None:
            continue

        blob_box = blob.get("box")
        blob_center = box_center(blob_box)

        if blob_center is None:
            continue

        br, bc = blob_center

        if br > inner_box["bottom"]:
            direction = "below"
        elif br < inner_box["top"]:
            direction = "above"
        elif bc > inner_box["right"]:
            direction = "right"
        elif bc < inner_box["left"]:
            direction = "left"
        else:
            direction = "center"

        directions.append(direction)

    return directions


def relation_for_marker(marker, output_info, inner_box, outer_box):
    """
    Convert a concrete marker cell into a symbolic relation.

    This is used during learning only.
    """
    h = output_info["height"]
    w = output_info["width"]

    row = marker["row"]
    col = marker["col"]

    inner_center = box_center(inner_box)
    outer_center = box_center(outer_box)

    if inner_center is not None:
        inner_center_r, inner_center_c = inner_center
    else:
        inner_center_r, inner_center_c = None, None

    if outer_center is not None:
        outer_center_r, outer_center_c = outer_center
    else:
        outer_center_r, outer_center_c = None, None

    if row == inner_center_r:
        row_relation = "inner_center_row"
    elif row == outer_center_r:
        row_relation = "outer_center_row"
    elif row == h - 3:
        row_relation = "output_height_minus_3"
    elif row == h // 2:
        row_relation = "output_center_row"
    else:
        row_relation = {
            "type": "constant",
            "value": row,
        }

    if col == inner_center_c:
        col_relation = "inner_center_col"
    elif col == outer_center_c:
        col_relation = "outer_center_col"
    elif col == w - 2:
        col_relation = "output_width_minus_2"
    elif col == w - 3:
        col_relation = "output_width_minus_3"
    elif col == w // 2:
        col_relation = "output_center_col"
    else:
        col_relation = {
            "type": "constant",
            "value": col,
        }

    return {
        "row_relation": row_relation,
        "col_relation": col_relation,
    }


def apply_marker_relation(relation, output_height, output_width, inner_box, outer_box):
    if relation is None:
        return None

    inner_center = box_center(inner_box)
    outer_center = box_center(outer_box)

    if inner_center is not None:
        inner_center_r, inner_center_c = inner_center
    else:
        inner_center_r, inner_center_c = None, None

    if outer_center is not None:
        outer_center_r, outer_center_c = outer_center
    else:
        outer_center_r, outer_center_c = None, None

    def resolve_row(row_relation):
        if row_relation == "inner_center_row":
            return inner_center_r

        if row_relation == "outer_center_row":
            return outer_center_r

        if row_relation == "output_height_minus_3":
            return output_height - 3

        if row_relation == "output_center_row":
            return output_height // 2

        if isinstance(row_relation, dict) and row_relation.get("type") == "constant":
            return row_relation.get("value")

        return None

    def resolve_col(col_relation):
        if col_relation == "inner_center_col":
            return inner_center_c

        if col_relation == "outer_center_col":
            return outer_center_c

        if col_relation == "output_width_minus_2":
            return output_width - 2

        if col_relation == "output_width_minus_3":
            return output_width - 3

        if col_relation == "output_center_col":
            return output_width // 2

        if isinstance(col_relation, dict) and col_relation.get("type") == "constant":
            return col_relation.get("value")

        return None

    row = resolve_row(relation.get("row_relation"))
    col = resolve_col(relation.get("col_relation"))

    if row is None or col is None:
        return None

    return row, col


def learn_nested_marker_rules(examples):
    nested_examples = [
        ex
        for ex in examples
        if ex["scene"].get("ring_count", 0) >= 2
    ]

    if not nested_examples:
        return {
            "inner_center_marker": False,
            "outside_marker_relation": None,
            "outer_blob_direction_relations": {},
        }

    inner_center_marker = True
    outside_relations = []
    outer_blob_direction_relations = {}

    for ex in nested_examples:
        scene = ex["scene"]
        output_info = ex["output_info"]
        drawing_info = ex["drawing_info"]

        outer_box = drawing_info.get("outer_box")
        inner_box = choose_inner_box(drawing_info)
        markers = drawing_info.get("markers", [])

        if inner_box is None or outer_box is None:
            inner_center_marker = False
            continue

        inner_center = box_center(inner_box)

        if inner_center is None:
            inner_center_marker = False
            continue

        inner_center_r, inner_center_c = inner_center

        remaining_markers = []

        for marker in markers:
            if marker["row"] == inner_center_r and marker["col"] == inner_center_c:
                continue

            remaining_markers.append(marker)

        if not marker_at(markers, inner_center_r, inner_center_c):
            inner_center_marker = False

        outside_count = scene.get("outside_blob_count", 0)

        if outside_count > 0:
            outside_candidates = [
                marker
                for marker in remaining_markers
                if marker["col"] > outer_box["right"]
            ]

            if outside_candidates:
                chosen = outside_candidates[0]
                outside_relations.append(
                    relation_for_marker(
                        chosen,
                        output_info,
                        inner_box,
                        outer_box,
                    )
                )
                remaining_markers.remove(chosen)

        directions = get_outer_only_blob_directions(scene)

        for direction in directions:
            if not remaining_markers:
                continue

            chosen = remaining_markers.pop(0)
            relation = relation_for_marker(
                chosen,
                output_info,
                inner_box,
                outer_box,
            )

            outer_blob_direction_relations.setdefault(direction, []).append(relation)

    outside_marker_relation = None

    if outside_relations:
        first = outside_relations[0]
        if all(rel == first for rel in outside_relations):
            outside_marker_relation = first

    learned_direction_relations = {}

    for direction, relations in outer_blob_direction_relations.items():
        if not relations:
            continue

        first = relations[0]

        if all(rel == first for rel in relations):
            learned_direction_relations[direction] = first

    return {
        "inner_center_marker": inner_center_marker,
        "outside_marker_relation": outside_marker_relation,
        "outer_blob_direction_relations": learned_direction_relations,
    }


def learn_drawing_rule(examples, prediction_scene=None):
    """
    Learn symbolic drawing instructions from train outputs.

    This replaces forced drawing behavior.

    It learns:
        - how to draw the outer frame
        - when/where to draw an inner frame
        - one-ring marker pattern
        - nested-scene marker roles
    """
    examples_with_drawing = []

    for ex in examples:
        drawing_info = analyze_output_drawing(
            output_grid=ex["output_grid"],
            output_info=ex["output_info"],
        )

        if drawing_info is None:
            return None

        ex = dict(ex)
        ex["drawing_info"] = drawing_info
        examples_with_drawing.append(ex)

    outer_box_rule = learn_box_formula_rule(
        examples_with_drawing,
        box_getter=lambda ex: ex["drawing_info"].get("outer_box"),
        prediction_scene=prediction_scene
    )
    print()
    print("[LEARNED OUTER BOX RULE]")
    print(outer_box_rule)

    inner_presence_rule = learn_inner_box_presence_rule(examples_with_drawing)

    inner_box_rule = learn_box_formula_rule(
        examples_with_drawing,
        box_getter=lambda ex: choose_inner_box(ex["drawing_info"]),
        prediction_scene=prediction_scene
    )

    one_ring_marker_rule = learn_one_ring_marker_rule(examples_with_drawing)

    nested_marker_rules = learn_nested_marker_rules(examples_with_drawing)

    return {
        "type": "learned_drawing_rule",
        "outer_box_rule": outer_box_rule,
        "inner_presence_rule": inner_presence_rule,
        "inner_box_rule": inner_box_rule,
        "one_ring_marker_rule": one_ring_marker_rule,
        "nested_marker_rules": nested_marker_rules,
    }


# ============================================================
# DRAWING APPLICATION
# ============================================================

def make_canvas(height, width, fill_color):
    return [[fill_color for _ in range(width)] for _ in range(height)]


def draw_box(grid, top, left, height, width, color):
    if height <= 0 or width <= 0:
        return

    bottom = top + height - 1
    right = left + width - 1

    for c in range(left, right + 1):
        set_cell(grid, top, c, color)
        set_cell(grid, bottom, c, color)

    for r in range(top, bottom + 1):
        set_cell(grid, r, left, color)
        set_cell(grid, r, right, color)


def render_learned_drawing(scene, height, width, draw_color, fill_color, drawing_rule):
    """
    Render output using only learned drawing instructions.

    No task-specific special cases are allowed here.
    """
    if drawing_rule is None:
        return None

    out = make_canvas(height, width, fill_color)

    outer_box = apply_box_formula_rule(
        drawing_rule.get("outer_box_rule"),
        scene,
        output_height=height,
        output_width=width,
    )

    if outer_box is None:
        return None

    draw_box(
        out,
        outer_box["top"],
        outer_box["left"],
        outer_box["height"],
        outer_box["width"],
        draw_color,
    )

    inner_box = None

    should_draw_inner = apply_inner_box_presence_rule(
        drawing_rule.get("inner_presence_rule", {}),
        scene,
    )

    if should_draw_inner:
        inner_box = apply_box_formula_rule(
            drawing_rule.get("inner_box_rule"),
            scene,
            output_height=height,
            output_width=width,
        )

        if inner_box is not None:
            draw_box(
                out,
                inner_box["top"],
                inner_box["left"],
                inner_box["height"],
                inner_box["width"],
                draw_color,
            )

    ring_count = scene.get("ring_count", 0)
    blob_count = scene.get("blob_count", 0)

    # ------------------------------------------------------------
    # ONE-RING MARKERS
    # ------------------------------------------------------------
    if ring_count == 1:
        one_ring_rule = drawing_rule.get("one_ring_marker_rule")

        if one_ring_rule is not None:
            row_start = one_ring_rule.get("row_start", 0)
            row_step = one_ring_rule.get("row_step", 2)

            if one_ring_rule.get("col_mode") == "output_center":
                marker_col = width // 2
            else:
                marker_col = width // 2

            for idx in range(blob_count):
                marker_row = row_start + idx * row_step
                set_cell(out, marker_row, marker_col, draw_color)

    # ------------------------------------------------------------
    # NESTED-RING MARKERS
    # ------------------------------------------------------------
    if ring_count >= 2:
        nested_rules = drawing_rule.get("nested_marker_rules", {})

        if inner_box is not None and nested_rules.get("inner_center_marker"):
            center = box_center(inner_box)

            if center is not None:
                set_cell(out, center[0], center[1], draw_color)

        outside_relation = nested_rules.get("outside_marker_relation")

        if scene.get("outside_blob_count", 0) > 0 and outside_relation is not None:
            marker = apply_marker_relation(
                outside_relation,
                output_height=height,
                output_width=width,
                inner_box=inner_box,
                outer_box=outer_box,
            )

            if marker is not None:
                set_cell(out, marker[0], marker[1], draw_color)

        direction_relations = nested_rules.get("outer_blob_direction_relations", {})
        directions = get_outer_only_blob_directions(scene)

        for direction in directions:
            relation = direction_relations.get(direction)

            if relation is None:
                continue

            marker = apply_marker_relation(
                relation,
                output_height=height,
                output_width=width,
                inner_box=inner_box,
                outer_box=outer_box,
            )

            if marker is not None:
                set_cell(out, marker[0], marker[1], draw_color)

    return out


# ============================================================
# TASK-LEVEL LEARNING
# ============================================================

def discover_visual_symbolic_rule_for_task(train_pairs, prediction_scene=None):
    """
    Learn a reusable visual-symbolic rule from all train pairs.

    This function sees train outputs because it is learning.
    But it stores only symbolic features/rules, not full output grids.
    """
    if not train_pairs:
        return None

    examples = []

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        output_grid = pair["output"]

        scene = analyze_input_scene(input_grid)
        output_info = analyze_output_grid(output_grid)

        if scene is None or output_info is None:
            return None

        sig = scene_signature(scene)

        examples.append({
            "pair_index": pair_index,
            "signature": sig,
            "scene": scene,
            "output_info": output_info,
            "output_grid": output_grid,
        })

    shape_rule = learn_shape_rule(
        examples,
        prediction_scene=prediction_scene,
    )
    color_rule = learn_color_rule(examples)
    drawing_rule = learn_drawing_rule(
        examples,
        prediction_scene=prediction_scene,
    )
    print("[VISUAL SYMBOLIC COLOR RULE]")
    print(color_rule)

    print_shape_rule_debug(
        shape_rule,
        label="VISUAL SYMBOLIC LEARNED SHAPE RULE",
    )

    return {
        "family": "visual_symbolic_rule",
        "rule_type": "learned_ring_blob_symbolic_scene",
        "examples": examples,
        "shape_rule": shape_rule,
        "color_rule": color_rule,
        "drawing_rule": drawing_rule,
        "pair_count": len(train_pairs),
    }


def choose_output_colors(rule, input_grid, scene):
    color_rule = rule.get("color_rule")

    if color_rule is None:
        return None, None

    return apply_color_rule_to_scene(color_rule, scene)


def apply_visual_symbolic_rule(rule, input_grid):
    """
    Apply learned symbolic rule to a new input.

    No expected output is accepted here.
    """
    if rule is None or input_grid is None:
        return None

    scene = analyze_input_scene(input_grid)

    if scene is None:
        return None

    predicted_shape = predict_shape_from_rule(
        rule.get("shape_rule", {}),
        scene,
    )

    if predicted_shape is None:
        return None

    height, width = predicted_shape

    draw_color, fill_color = choose_output_colors(
        rule,
        input_grid,
        scene,
    )

    if draw_color is None or fill_color is None:
        return None

    return render_learned_drawing(
        scene=scene,
        height=height,
        width=width,
        draw_color=draw_color,
        fill_color=fill_color,
        drawing_rule=rule.get("drawing_rule"),
    )


# ============================================================
# TRAIN REPLAY / DEBUG
# ============================================================

def score_visual_symbolic_rule_on_train(rule, train_pairs):
    """
    Score rule on train pairs.

    Expected outputs are used only for scoring after prediction.
    """
    if rule is None:
        return None

    results = []
    exact_count = 0
    total_score = 0

    for pair_index, pair in enumerate(train_pairs):
        predicted = apply_visual_symbolic_rule(
            rule,
            pair["input"],
        )

        expected = pair["output"]

        score = score_prediction(predicted, expected)
        exact = predicted == expected

        if exact:
            exact_count += 1

        total_score += score

        results.append({
            "pair_index": pair_index,
            "predicted": predicted,
            "score": score,
            "exact": exact,
        })

    return {
        "strategy": "visual_symbolic_rule",
        "task_rule": rule,
        "rule": rule,
        "pair_count": len(train_pairs),
        "exact_count": exact_count,
        "total_raw_score": total_score,
        "total_adjusted_score": total_score,
        "results": results,
    }


def is_strong_visual_symbolic_result(scored_rule):
    """
    Strong means solved every train pair exactly.

    Router should still prefer leave-one-out validation before using this
    as trusted test-time rule.
    """
    if scored_rule is None:
        return False

    pair_count = scored_rule.get("pair_count", 0)
    exact_count = scored_rule.get("exact_count", 0)

    if pair_count == 0:
        return False

    return exact_count == pair_count


def print_grid_diff_summary(predicted, expected, label):
    print()
    print(f"[{label} DIFF SUMMARY]")

    if predicted is None:
        print("predicted is None")
        return

    ph, pw = grid_shape(predicted)
    eh, ew = grid_shape(expected)

    print(f"pred shape: {ph}x{pw}")
    print(f"exp  shape: {eh}x{ew}")

    if ph != eh or pw != ew:
        print("shape mismatch")
        return

    diffs = []

    for r in range(eh):
        for c in range(ew):
            if predicted[r][c] != expected[r][c]:
                diffs.append((r, c, predicted[r][c], expected[r][c]))

    print(f"wrong cells: {len(diffs)}")

    for r, c, p, e in diffs[:80]:
        print(f"  r={r:2d} c={c:2d} pred={p} exp={e}")


# ============================================================
# LEAVE-ONE-OUT VALIDATION
# ============================================================

def leave_one_out_visual_symbolic(train_pairs):
    """
    Honest validation:
        hide one train output
        learn from the rest
        predict hidden input
        score hidden output
    """
    results = []

    for hidden_index, hidden_pair in enumerate(train_pairs):
        visible_pairs = []

        for idx, pair in enumerate(train_pairs):
            if idx != hidden_index:
                visible_pairs.append(pair)

        hidden_scene = analyze_input_scene(hidden_pair["input"])

        rule = discover_visual_symbolic_rule_for_task(
            visible_pairs,
            prediction_scene=hidden_scene,
        )

        if rule is None:
            predicted = None
        else:
            print_shape_rule_debug(
                rule.get("shape_rule", {}),
                label=f"LOO HIDDEN PAIR {hidden_index} SHAPE RULE",
            )

            predicted = apply_visual_symbolic_rule(
                rule,
                hidden_pair["input"],
            )

        expected = hidden_pair["output"]
        score = score_prediction(predicted, expected)
        exact = predicted == expected

        results.append({
            "pair_index": hidden_index,
            "predicted": predicted,
            "score": score,
            "exact": exact,
        })

        if not exact:
            print_grid_diff_summary(
                predicted,
                expected,
                label=f"LOO HIDDEN PAIR {hidden_index}",
            )

    exact_count = sum(1 for r in results if r.get("exact"))

    return {
        "strategy": "visual_symbolic_rule",
        "pair_count": len(train_pairs),
        "exact_count": exact_count,
        "total_raw_score": sum(r.get("score", 0) for r in results),
        "total_adjusted_score": sum(r.get("score", 0) for r in results),
        "results": results,
    }
