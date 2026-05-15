# reasoning/visual_abstraction_mapper.py
"""
Visual Abstraction Mapper

Purpose:
    Learn how input visual abstractions map to output abstractions/templates.

Important:
    This file should NOT hardcode task-specific drawing logic.
    This file should NOT print debug output.
    This file should NOT open popup windows.

This file is the learner/mapper brain.

The debugger should handle:
    - printing
    - popups
    - visual comparisons
    - action logs
    - delta displays

Main public functions:
    learn_visual_abstraction_mapping(...)
    learn_assignment_marker_growth_rules(...)
    choose_closest_mapping_for_input(...)
    create_prediction_from_closest_mapping(...)
    create_adapted_prediction_from_closest_mapping(...)
"""

from collections import Counter, deque


# ============================================================
# Import visual abstraction discovery
# ============================================================

try:
    from reasoning.visual_abstraction_discovery import discover_visual_abstractions
except ImportError:
    from visual_abstraction_discovery import discover_visual_abstractions


# ============================================================
# Basic grid helpers
# ============================================================

def grid_shape(grid):
    if not grid:
        return (0, 0)

    return (len(grid), len(grid[0]))


def copy_grid(grid):
    return [row[:] for row in grid]


def in_bounds(grid, r, c):
    h, w = grid_shape(grid)
    return 0 <= r < h and 0 <= c < w


def get_nonzero_color_counts(grid):
    counts = Counter()

    for row in grid:
        for value in row:
            if value != 0:
                counts[value] += 1

    return counts


def most_common_nonzero_color(grid):
    counts = get_nonzero_color_counts(grid)

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


# ============================================================
# Connected component helpers
# ============================================================

def find_components_by_predicate(grid, predicate):
    """
    Find connected components where predicate(value, r, c) is True.

    Uses 4-connectivity.
    """
    h, w = grid_shape(grid)
    visited = set()
    components = []

    for r in range(h):
        for c in range(w):
            if (r, c) in visited:
                continue

            value = grid[r][c]

            if not predicate(value, r, c):
                continue

            q = deque([(r, c)])
            visited.add((r, c))
            cells = []

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc, grid[cr][cc]))

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = cr + dr
                    nc = cc + dc

                    if not in_bounds(grid, nr, nc):
                        continue

                    if (nr, nc) in visited:
                        continue

                    if not predicate(grid[nr][nc], nr, nc):
                        continue

                    visited.add((nr, nc))
                    q.append((nr, nc))

            components.append(component_summary(cells))

    return components


def component_summary(cells):
    if not cells:
        return {
            "cells": [],
            "cell_count": 0,
            "bbox": None,
            "height": 0,
            "width": 0,
            "colors": [],
            "color_counts": {},
        }

    rows = [r for r, c, value in cells]
    cols = [c for r, c, value in cells]
    values = [value for r, c, value in cells]

    min_r = min(rows)
    max_r = max(rows)
    min_c = min(cols)
    max_c = max(cols)

    color_counts = Counter(values)

    return {
        "cells": cells,
        "cell_count": len(cells),
        "bbox": (min_r, min_c, max_r, max_c),
        "height": max_r - min_r + 1,
        "width": max_c - min_c + 1,
        "colors": sorted(color_counts.keys()),
        "color_counts": dict(color_counts),
    }


# ============================================================
# View extraction helpers
# ============================================================

def normalize_views(raw_views):
    """
    Normalize whatever visual_abstraction_discovery returns into a list of views.
    """
    if raw_views is None:
        return []

    if isinstance(raw_views, list):
        return raw_views

    if isinstance(raw_views, dict):
        if "views" in raw_views and isinstance(raw_views["views"], list):
            return raw_views["views"]

        result = []

        for key, value in raw_views.items():
            if isinstance(value, dict):
                view = dict(value)
                view.setdefault("view_type", key)
                result.append(view)

        return result

    return []


def get_view_by_type(grid, preferred_view_type):
    raw_views = discover_visual_abstractions(grid)
    views = normalize_views(raw_views)

    for view in views:
        if view.get("view_type") == preferred_view_type:
            return view

    if views:
        return views[0]

    return {
        "view_type": preferred_view_type,
        "ring_count": 0,
        "blob_count": 0,
        "assignments": [],
        "rings": [],
        "blobs": [],
    }


# ============================================================
# Box helpers
# ============================================================

def get_box_from_item(item):
    """
    Accept several possible box formats.

    Returns:
        (min_r, min_c, max_r, max_c) or None
    """
    if item is None:
        return None

    if isinstance(item, dict):
        for key in ["bbox", "box", "bounds"]:
            if key in item:
                return get_box_from_item(item[key])

        if all(k in item for k in ["min_r", "min_c", "max_r", "max_c"]):
            return (
                item["min_r"],
                item["min_c"],
                item["max_r"],
                item["max_c"],
            )

        if all(k in item for k in ["top", "left", "bottom", "right"]):
            return (
                item["top"],
                item["left"],
                item["bottom"],
                item["right"],
            )

    if isinstance(item, (list, tuple)) and len(item) == 4:
        return tuple(item)

    return None


def box_height(box):
    if box is None:
        return 0

    return box[2] - box[0] + 1


def box_width(box):
    if box is None:
        return 0

    return box[3] - box[1] + 1


def box_area(box):
    if box is None:
        return 0

    return box_height(box) * box_width(box)


def box_center(box):
    if box is None:
        return (0.0, 0.0)

    return (
        (box[0] + box[2]) / 2.0,
        (box[1] + box[3]) / 2.0,
    )


def normalize_box(box, grid_h, grid_w):
    if box is None:
        return None

    max_r = max(1, grid_h - 1)
    max_c = max(1, grid_w - 1)

    return (
        round(box[0] / max_r, 3),
        round(box[1] / max_c, 3),
        round(box[2] / max_r, 3),
        round(box[3] / max_c, 3),
    )


# ============================================================
# Assignment parsing
# ============================================================

def normalize_assignment_label(raw_label):
    """
    Convert assignment labels into stable text.

    Examples:
        None          -> outside_all
        "outside"     -> outside_all
        0             -> ring_0
        1             -> ring_1
        "ring 0"      -> ring_0
        "ring_0"      -> ring_0
    """
    if raw_label is None:
        return "outside_all"

    if isinstance(raw_label, int):
        if raw_label < 0:
            return "outside_all"

        return f"ring_{raw_label}"

    label = str(raw_label).strip()

    if label in ["outside", "outside_all", "none", "None", "-1"]:
        return "outside_all"

    label = label.replace(" ", "_")

    if label.startswith("ring_"):
        return label

    if label.startswith("ring"):
        suffix = label.replace("ring", "").replace("_", "")

        if suffix.isdigit():
            return f"ring_{suffix}"

    if label.isdigit():
        return f"ring_{label}"

    return label


def extract_assignment_label(assignment):
    """
    Defensive parser for assignment records.

    Possible formats:
        {"blob_id": 2, "ring_id": 0}
        {"blob": 2, "assigned_to": "ring_0"}
        (2, 0)
        (2, "outside_all")
    """
    if isinstance(assignment, dict):
        for key in [
            "assigned_to",
            "target",
            "ring",
            "ring_id",
            "ring_index",
            "container",
            "parent",
        ]:
            if key in assignment:
                return normalize_assignment_label(assignment[key])

        return "outside_all"

    if isinstance(assignment, (list, tuple)):
        if len(assignment) >= 2:
            return normalize_assignment_label(assignment[1])

    return normalize_assignment_label(assignment)


def normalize_assignment_signature(assignments):
    labels = [extract_assignment_label(a) for a in assignments]

    def sort_key(label):
        if label == "outside_all":
            return (-1, label)

        if label.startswith("ring_"):
            suffix = label.replace("ring_", "")

            if suffix.isdigit():
                return (int(suffix), label)

        return (999, label)

    return tuple(sorted(labels, key=sort_key))


# ============================================================
# Input abstraction extraction
# ============================================================

def extract_input_features(input_grid, preferred_view_type="ring_blob_view"):
    """
    Extract task-level input features from the selected visual view.

    This describes what the solver sees.
    It does not solve.
    """
    h, w = grid_shape(input_grid)

    view = get_view_by_type(
        input_grid,
        preferred_view_type=preferred_view_type,
    )

    rings = (
        view.get("rings", [])
        or view.get("ring_components", [])
        or []
    )

    blobs = (
        view.get("blobs", [])
        or view.get("blob_components", [])
        or []
    )

    assignments = (
        view.get("assignments", [])
        or view.get("blob_assignments", [])
        or []
    )

    ring_count = view.get("ring_count", len(rings))
    blob_count = view.get("blob_count", len(blobs))

    assignment_signature = normalize_assignment_signature(assignments)

    if not assignment_signature and blob_count > 0:
        assignment_signature = tuple(["unknown"] * blob_count)

    outside_blob_count = sum(
        1 for label in assignment_signature
        if label == "outside_all"
    )

    inside_blob_count = max(0, blob_count - outside_blob_count)

    ring_boxes = []

    for ring in rings:
        box = get_box_from_item(ring)

        ring_boxes.append({
            "bbox": box,
            "height": box_height(box),
            "width": box_width(box),
            "area": box_area(box),
            "center": box_center(box),
            "normalized_bbox": normalize_box(box, h, w),
        })

    blob_boxes = []

    for blob in blobs:
        box = get_box_from_item(blob)

        blob_boxes.append({
            "bbox": box,
            "height": box_height(box),
            "width": box_width(box),
            "area": box_area(box),
            "center": box_center(box),
            "normalized_bbox": normalize_box(box, h, w),
        })

    assignment_counts = Counter(assignment_signature)

    return {
        "view_type": view.get("view_type", preferred_view_type),
        "grid_shape": (h, w),

        "ring_count": ring_count,
        "blob_count": blob_count,
        "inside_blob_count": inside_blob_count,
        "outside_blob_count": outside_blob_count,

        "assignment_signature": assignment_signature,
        "assignment_counts": dict(assignment_counts),

        "ring_boxes": ring_boxes,
        "blob_boxes": blob_boxes,

        "has_outside_blob": outside_blob_count > 0,
        "has_multiple_rings": ring_count >= 2,
        "has_multiple_blobs": blob_count >= 2,
    }


def make_input_signature(input_features):
    """
    Stable signature used for coarse task-level matching.
    """
    return (
        input_features.get("ring_count", 0),
        input_features.get("blob_count", 0),
        input_features.get("outside_blob_count", 0),
        input_features.get("inside_blob_count", 0),
        tuple(input_features.get("assignment_signature", ())),
    )


# ============================================================
# Output abstraction extraction
# ============================================================

def get_border_positions(grid):
    h, w = grid_shape(grid)
    positions = []

    if h == 0 or w == 0:
        return positions

    for c in range(w):
        positions.append((0, c))

        if h > 1:
            positions.append((h - 1, c))

    for r in range(1, h - 1):
        positions.append((r, 0))

        if w > 1:
            positions.append((r, w - 1))

    return unique_in_order(positions)


def get_border_color(output_grid):
    border_positions = get_border_positions(output_grid)
    counts = Counter()

    for r, c in border_positions:
        value = output_grid[r][c]

        if value != 0:
            counts[value] += 1

    if counts:
        return counts.most_common(1)[0][0]

    return most_common_nonzero_color(output_grid)


def get_fill_color(output_grid):
    h, w = grid_shape(output_grid)
    border_color = get_border_color(output_grid)
    interior_counts = Counter()

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            value = output_grid[r][c]

            if value != 0 and value != border_color:
                interior_counts[value] += 1

    if interior_counts:
        return interior_counts.most_common(1)[0][0]

    counts = get_nonzero_color_counts(output_grid)

    if border_color in counts:
        del counts[border_color]

    if counts:
        return counts.most_common(1)[0][0]

    return border_color


def output_border_ratio(output_grid):
    if not output_grid:
        return 0.0

    border_color = get_border_color(output_grid)
    positions = get_border_positions(output_grid)

    if not positions:
        return 0.0

    matching = 0

    for r, c in positions:
        if output_grid[r][c] == border_color:
            matching += 1

    return round(matching / max(1, len(positions)), 3)


def estimate_frame_depth(output_grid):
    """
    Estimate how many mostly-complete frame layers exist.
    """
    h, w = grid_shape(output_grid)

    if h == 0 or w == 0:
        return 0

    border_color = get_border_color(output_grid)
    depth = 0
    max_layers = min(h, w) // 2 + 1

    for layer in range(max_layers):
        top = layer
        left = layer
        bottom = h - 1 - layer
        right = w - 1 - layer

        if top > bottom or left > right:
            break

        positions = []

        for c in range(left, right + 1):
            positions.append((top, c))

            if bottom != top:
                positions.append((bottom, c))

        for r in range(top + 1, bottom):
            positions.append((r, left))

            if right != left:
                positions.append((r, right))

        positions = unique_in_order(positions)

        if not positions:
            break

        match_count = sum(
            1 for r, c in positions
            if output_grid[r][c] == border_color
        )

        ratio = match_count / max(1, len(positions))

        if ratio >= 0.80:
            depth += 1
        else:
            break

    return depth


def is_border_cell(grid, r, c):
    h, w = grid_shape(grid)
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def is_marker_cell(output_grid, r, c, border_color, fill_color):
    """
    Marker cells are meaningful interior cells that are not simple fill.
    """
    value = output_grid[r][c]

    if value == 0:
        return False

    if is_border_cell(output_grid, r, c):
        return False

    if value == border_color:
        return True

    if value != fill_color:
        return True

    return False


def extract_marker_cells(output_grid):
    h, w = grid_shape(output_grid)
    border_color = get_border_color(output_grid)
    fill_color = get_fill_color(output_grid)
    marker_cells = []

    for r in range(h):
        for c in range(w):
            if is_marker_cell(output_grid, r, c, border_color, fill_color):
                marker_cells.append({
                    "r": r,
                    "c": c,
                    "color": output_grid[r][c],
                    "r_ratio": round(r / max(1, h - 1), 3),
                    "c_ratio": round(c / max(1, w - 1), 3),
                })

    return marker_cells


def extract_marker_groups(output_grid):
    border_color = get_border_color(output_grid)
    fill_color = get_fill_color(output_grid)

    components = find_components_by_predicate(
        output_grid,
        lambda value, r, c: is_marker_cell(
            output_grid,
            r,
            c,
            border_color,
            fill_color,
        ),
    )

    h, w = grid_shape(output_grid)
    groups = []

    for idx, comp in enumerate(components):
        box = comp.get("bbox")

        groups.append({
            "group_index": idx,
            "cell_count": comp.get("cell_count", 0),
            "bbox": box,
            "height": box_height(box),
            "width": box_width(box),
            "area": box_area(box),
            "normalized_bbox": normalize_box(box, h, w),
            "colors": comp.get("colors", []),
            "color_counts": comp.get("color_counts", {}),
        })

    return groups


def count_interior_frame_cells(output_grid):
    """
    Small readability wrapper.
    Kept for now because older code may expect this name.
    """
    return len(extract_marker_cells(output_grid))


def extract_output_template(output_grid):
    """
    Capture a train output as a reusable template.
    """
    h, w = grid_shape(output_grid)

    border_color = get_border_color(output_grid)
    fill_color = get_fill_color(output_grid)
    frame_depth = estimate_frame_depth(output_grid)
    marker_cells = extract_marker_cells(output_grid)
    marker_groups = extract_marker_groups(output_grid)

    return {
        "shape": (h, w),
        "border_color": border_color,
        "fill_color": fill_color,
        "border_ratio": output_border_ratio(output_grid),
        "frame_depth": frame_depth,

        "marker_count": len(marker_cells),
        "marker_cells": marker_cells,
        "marker_groups": marker_groups,
        "marker_group_count": len(marker_groups),

        "raw_output_grid": copy_grid(output_grid),
    }


def extract_output_abstraction(output_grid):
    template = extract_output_template(output_grid)

    return {
        "shape": template["shape"],
        "border_color": template["border_color"],
        "fill_color": template["fill_color"],
        "border_ratio": template["border_ratio"],
        "frame_depth": template["frame_depth"],
        "interior_frame_cells": template["marker_count"],
        "marker_group_count": template["marker_group_count"],
    }


# ============================================================
# Pair mapping learner
# ============================================================

def learn_pair_mapping(pair, pair_index, preferred_view_type="ring_blob_view"):
    input_grid = pair["input"]
    output_grid = pair["output"]

    input_features = extract_input_features(
        input_grid,
        preferred_view_type=preferred_view_type,
    )

    output_abstraction = extract_output_abstraction(output_grid)
    output_template = extract_output_template(output_grid)
    input_signature = make_input_signature(input_features)

    return {
        "pair_index": pair_index,
        "preferred_view_type": preferred_view_type,

        "input_signature": input_signature,
        "input_features": input_features,

        "output_abstraction": output_abstraction,
        "output_template": output_template,

        "learned_relation": {
            "rings_to_frame_depth": (
                input_features.get("ring_count", 0),
                output_abstraction.get("frame_depth", 0),
            ),
            "blobs_to_marker_cells": (
                input_features.get("blob_count", 0),
                output_abstraction.get("interior_frame_cells", 0),
            ),
            "blobs_to_marker_groups": (
                input_features.get("blob_count", 0),
                output_abstraction.get("marker_group_count", 0),
            ),
            "outside_blobs_to_output_shape": (
                input_features.get("outside_blob_count", 0),
                output_abstraction.get("shape", (0, 0)),
            ),
        },
    }


def build_signature_to_outputs(pair_mappings):
    """
    Build a lookup table from input signatures to learned mappings.

    This is not heavily used yet, but it is useful for future exact-signature
    matching or debugging.
    """
    table = {}

    for mapping in pair_mappings:
        sig = mapping["input_signature"]
        table.setdefault(sig, [])
        table[sig].append(mapping)

    return table


def learn_visual_abstraction_mapping(
    train_pairs,
    preferred_view_type="ring_blob_view",
):
    pair_mappings = []

    for idx, pair in enumerate(train_pairs, start=1):
        mapping = learn_pair_mapping(
            pair,
            pair_index=idx,
            preferred_view_type=preferred_view_type,
        )

        pair_mappings.append(mapping)

    output_shapes = unique_in_order(
        mapping["output_abstraction"]["shape"]
        for mapping in pair_mappings
    )

    frame_depths = unique_in_order(
        mapping["output_abstraction"]["frame_depth"]
        for mapping in pair_mappings
    )

    ring_counts = unique_in_order(
        mapping["input_features"]["ring_count"]
        for mapping in pair_mappings
    )

    blob_counts = unique_in_order(
        mapping["input_features"]["blob_count"]
        for mapping in pair_mappings
    )

    assignment_signatures = unique_in_order(
        mapping["input_features"]["assignment_signature"]
        for mapping in pair_mappings
    )

    marker_counts = unique_in_order(
        mapping["output_template"]["marker_count"]
        for mapping in pair_mappings
    )

    marker_group_counts = unique_in_order(
        mapping["output_template"]["marker_group_count"]
        for mapping in pair_mappings
    )

    return {
        "family": "visual_abstraction_mapping",
        "preferred_view_type": preferred_view_type,
        "pair_count": len(pair_mappings),

        "pair_mappings": pair_mappings,
        "signature_to_outputs": build_signature_to_outputs(pair_mappings),

        "known_patterns": {
            "output_shapes": output_shapes,
            "frame_depths": frame_depths,
            "ring_counts": ring_counts,
            "blob_counts": blob_counts,
            "assignment_signatures": assignment_signatures,
            "marker_counts": marker_counts,
            "marker_group_counts": marker_group_counts,
        },
    }


# ============================================================
# Matching test input to learned mapping
# ============================================================

def multiset_overlap_score(a, b):
    ca = Counter(a)
    cb = Counter(b)

    score = 0
    all_keys = set(ca.keys()) | set(cb.keys())

    for key in all_keys:
        score += min(ca[key], cb[key]) * 8
        score -= abs(ca[key] - cb[key]) * 3

    return score


def score_mapping_for_input(input_features, mapping):
    """
    Score how well a train mapping matches a test input abstraction.

    Higher is better.
    """
    train_features = mapping["input_features"]

    score = 0

    ring_diff = abs(
        input_features.get("ring_count", 0)
        - train_features.get("ring_count", 0)
    )

    if ring_diff == 0:
        score += 50
    else:
        score -= 15 * ring_diff

    blob_diff = abs(
        input_features.get("blob_count", 0)
        - train_features.get("blob_count", 0)
    )

    if blob_diff == 0:
        score += 30
    else:
        score -= 8 * blob_diff

    outside_diff = abs(
        input_features.get("outside_blob_count", 0)
        - train_features.get("outside_blob_count", 0)
    )

    if outside_diff == 0:
        score += 40
    else:
        score -= 20 * outside_diff

    inside_diff = abs(
        input_features.get("inside_blob_count", 0)
        - train_features.get("inside_blob_count", 0)
    )

    if inside_diff == 0:
        score += 20
    else:
        score -= 6 * inside_diff

    test_sig = input_features.get("assignment_signature", ())
    train_sig = train_features.get("assignment_signature", ())

    score += multiset_overlap_score(test_sig, train_sig)

    if test_sig == train_sig:
        score += 100

    if input_features.get("has_outside_blob") == train_features.get("has_outside_blob"):
        score += 20
    else:
        score -= 20

    return score


def choose_closest_mapping_for_input(mapping_rule, input_grid):
    preferred_view_type = mapping_rule.get(
        "preferred_view_type",
        "ring_blob_view",
    )

    input_features = extract_input_features(
        input_grid,
        preferred_view_type=preferred_view_type,
    )

    input_signature = make_input_signature(input_features)

    best = None
    best_score = -10 ** 9

    for mapping in mapping_rule.get("pair_mappings", []):
        score = score_mapping_for_input(input_features, mapping)

        if score > best_score:
            best_score = score
            best = mapping

    if best is None:
        return None

    feature_delta = compute_feature_delta(
        input_features,
        best["input_features"],
    )

    return {
        "test_input_features": input_features,
        "test_signature": input_signature,

        "matched_mapping": best,
        "matched_train_pair": best["pair_index"],
        "match_score": best_score,

        "matched_train_input_features": best["input_features"],
        "feature_delta": feature_delta,

        "matched_output_abstraction": best["output_abstraction"],
        "matched_output_template": best["output_template"],
    }


# ============================================================
# Feature delta helpers
# ============================================================

def compute_feature_delta(test_input_features, matched_train_features):
    """
    Compare test input abstraction to matched train input abstraction.
    """
    test_assignment_counts = Counter(
        test_input_features.get("assignment_signature", ())
    )

    train_assignment_counts = Counter(
        matched_train_features.get("assignment_signature", ())
    )

    assignment_delta = {}
    all_assignment_keys = (
        set(test_assignment_counts.keys())
        | set(train_assignment_counts.keys())
    )

    for key in sorted(all_assignment_keys):
        assignment_delta[key] = (
            test_assignment_counts.get(key, 0)
            - train_assignment_counts.get(key, 0)
        )

    return {
        "ring_count_delta": (
            test_input_features.get("ring_count", 0)
            - matched_train_features.get("ring_count", 0)
        ),
        "blob_count_delta": (
            test_input_features.get("blob_count", 0)
            - matched_train_features.get("blob_count", 0)
        ),
        "inside_blob_count_delta": (
            test_input_features.get("inside_blob_count", 0)
            - matched_train_features.get("inside_blob_count", 0)
        ),
        "outside_blob_count_delta": (
            test_input_features.get("outside_blob_count", 0)
            - matched_train_features.get("outside_blob_count", 0)
        ),
        "assignment_delta": assignment_delta,
        "test_assignment_signature": test_input_features.get("assignment_signature", ()),
        "matched_train_assignment_signature": matched_train_features.get("assignment_signature", ()),
    }


# ============================================================
# Template prediction helpers
# ============================================================

def create_prediction_from_template(output_template):
    """
    Replay the closest learned output template.
    """
    return copy_grid(output_template["raw_output_grid"])


def create_prediction_from_closest_mapping(mapping_rule, input_grid):
    """
    Simple prediction:
        closest learned template replay.
    """
    match = choose_closest_mapping_for_input(mapping_rule, input_grid)

    if match is None:
        return None

    template = match["matched_output_template"]

    return create_prediction_from_template(template)


# ============================================================
# Marker helper functions
# ============================================================

def get_template_marker_color(output_template):
    """
    Choose the color used for learned marker additions.
    """
    marker_cells = output_template.get("marker_cells", [])

    if marker_cells:
        counts = Counter(cell["color"] for cell in marker_cells)
        return counts.most_common(1)[0][0]

    return output_template.get("border_color", 1)


def find_open_interior_cells(grid, fill_color):
    """
    Find interior cells that still match the learned fill color.
    """
    h, w = grid_shape(grid)
    cells = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r][c] == fill_color:
                cells.append((r, c))

    return cells


def distance_sq(a, b):
    ar, ac = a
    br, bc = b

    return (ar - br) ** 2 + (ac - bc) ** 2


def get_group_center(group):
    box = group.get("bbox")

    if box is None:
        return None

    return (
        (box[0] + box[2]) / 2.0,
        (box[1] + box[3]) / 2.0,
    )


def choose_anchor_group_for_assignment(output_template, assignment_label):
    """
    Pick a learned marker group as the anchor for an extra blob.
    """
    groups = output_template.get("marker_groups", [])

    if not groups:
        return None

    small_groups = [
        group for group in groups
        if group.get("cell_count", 0) <= 3
    ]

    candidate_groups = small_groups if small_groups else groups

    def group_sort_key(group):
        center = get_group_center(group)

        if center is None:
            return 9999

        return center[1]

    candidate_groups = sorted(candidate_groups, key=group_sort_key)

    if assignment_label == "ring_0":
        return candidate_groups[0]

    if assignment_label == "ring_1":
        return candidate_groups[-1]

    if assignment_label == "outside_all":
        return candidate_groups[-1]

    return candidate_groups[-1]


# ============================================================
# Assignment-level marker growth learning
# ============================================================

def get_marker_group_centers(output_template):
    """
    Return centers of small marker groups.
    """
    centers = []

    for group in output_template.get("marker_groups", []):
        if group.get("bbox") is None:
            continue

        if group.get("cell_count", 0) > 3:
            continue

        center = get_group_center(group)

        if center is None:
            continue

        centers.append({
            "center": (round(center[0]), round(center[1])),
            "group": group,
        })

    return centers


def count_assignment_instances(input_features):
    counts = Counter()

    for label in input_features.get("assignment_signature", ()):
        counts[label] += 1

    return counts


def infer_direction_from_centers(centers):
    """
    Infer simple growth direction from marker centers.
    """
    if len(centers) < 2:
        return {
            "direction": "unknown",
            "step": None,
            "centers": centers,
        }

    sorted_centers = sorted(centers)

    rows = [rc[0] for rc in sorted_centers]
    cols = [rc[1] for rc in sorted_centers]

    unique_rows = sorted(set(rows))
    unique_cols = sorted(set(cols))

    if len(unique_cols) == 1 and len(unique_rows) >= 2:
        row_diffs = [
            unique_rows[i + 1] - unique_rows[i]
            for i in range(len(unique_rows) - 1)
        ]

        step = max(set(row_diffs), key=row_diffs.count)

        return {
            "direction": "vertical",
            "step": step,
            "centers": sorted_centers,
        }

    if len(unique_rows) == 1 and len(unique_cols) >= 2:
        col_diffs = [
            unique_cols[i + 1] - unique_cols[i]
            for i in range(len(unique_cols) - 1)
        ]

        step = max(set(col_diffs), key=col_diffs.count)

        return {
            "direction": "horizontal",
            "step": step,
            "centers": sorted_centers,
        }

    return {
        "direction": "unknown",
        "step": None,
        "centers": sorted_centers,
    }


def learn_assignment_marker_growth_rules(mapping_rule):
    """
    Learn how markers grow for each assignment label.

    This looks across all train pair mappings.
    """
    pair_mappings = mapping_rule.get("pair_mappings", [])
    observations_by_assignment = {}

    for pair in pair_mappings:
        input_features = pair.get("input_features", {})
        output_template = pair.get("output_template", {})

        assignment_counts = count_assignment_instances(input_features)
        marker_centers = get_marker_group_centers(output_template)

        for assignment_label, assignment_count in assignment_counts.items():
            observations_by_assignment.setdefault(assignment_label, [])

            observations_by_assignment[assignment_label].append({
                "pair_index": pair.get("pair_index"),
                "assignment_count": assignment_count,
                "marker_centers": [item["center"] for item in marker_centers],
                "marker_center_count": len(marker_centers),
            })

    growth_rules = {}

    for assignment_label, observations in observations_by_assignment.items():
        observations = sorted(
            observations,
            key=lambda item: (
                item["assignment_count"],
                item["marker_center_count"],
            ),
        )

        best_rule = None

        for obs in observations:
            assignment_count = obs["assignment_count"]
            marker_centers = obs["marker_centers"]

            if assignment_count != len(marker_centers):
                continue

            direction_info = infer_direction_from_centers(marker_centers)

            if direction_info["direction"] == "unknown":
                continue

            best_rule = {
                "assignment": assignment_label,
                "source_pair": obs["pair_index"],
                "direction": direction_info["direction"],
                "step": direction_info["step"],
                "learned_centers": direction_info["centers"],
                "confidence": "clean_count_match",
            }

        if best_rule is not None:
            growth_rules[assignment_label] = best_rule

    mapping_rule["assignment_marker_growth_rules"] = growth_rules

    return growth_rules


def learn_generic_marker_growth_rule(mapping_rule):
    """
    Learn one generic task-level marker growth rule.

    Why this exists:
        Sometimes only one assignment has a clean direct growth example.

        Example:
            ring_0 has:
                1 marker -> 2 markers
                vertical step 2

            ring_1 does not have its own clean example.

        In that case, we can still learn a generic task behavior:

            repeated blobs create repeated markers
            using the same growth direction/step

    This is NOT hardcoding ring_1.
    It borrows from the strongest learned assignment rule.
    """
    assignment_rules = mapping_rule.get("assignment_marker_growth_rules", {})

    if not assignment_rules:
        mapping_rule["generic_marker_growth_rule"] = None
        return None

    # Prefer the rule with the strongest confidence.
    # For now, clean_count_match is our strongest signal.
    best_rule = None
    best_score = -10 ** 9

    for assignment_label, rule in assignment_rules.items():
        score = 0

        if rule.get("confidence") == "clean_count_match":
            score += 100

        if rule.get("direction") in ["vertical", "horizontal"]:
            score += 25

        if rule.get("step") is not None:
            score += 25

        # Prefer ring-specific rules over outside rules.
        if str(assignment_label).startswith("ring_"):
            score += 10

        if score > best_score:
            best_score = score
            best_rule = rule

    if best_rule is None:
        mapping_rule["generic_marker_growth_rule"] = None
        return None

    generic_rule = {
        "assignment": "generic_marker_growth",
        "source_assignment": best_rule.get("assignment"),
        "source_pair": best_rule.get("source_pair"),
        "direction": best_rule.get("direction"),
        "step": best_rule.get("step"),
        "learned_centers": best_rule.get("learned_centers"),
        "confidence": "borrowed_from_assignment_rule",
    }

    mapping_rule["generic_marker_growth_rule"] = generic_rule

    return generic_rule

# ============================================================
# Marker placement logic
# ============================================================

def choose_new_marker_location_from_growth_rule(
    grid,
    output_template,
    anchor_group,
    assignment_label,
    growth_rules,
    generic_growth_rule=None,
):
    """
    Try to place a new marker using learned growth behavior.

    Priority:
        1. Use assignment-specific rule.
            Example: ring_0 has its own rule.

        2. If missing, use generic task-level growth rule.
            Example: ring_1 borrows the learned ring_0 vertical-growth style.

        3. If both fail, return None and let local fallback handle it.
    """
    if growth_rules is None:
        growth_rules = {}

    rule = growth_rules.get(assignment_label)
    used_generic_rule = False

    if rule is None and generic_growth_rule is not None:
        rule = generic_growth_rule
        used_generic_rule = True

    if rule is None:
        return None

    fill_color = output_template.get("fill_color", 0)
    open_cells = set(find_open_interior_cells(grid, fill_color))

    if not open_cells:
        return None

    direction = rule.get("direction")
    step = rule.get("step")

    if direction not in ["vertical", "horizontal"]:
        return None

    if step is None or step <= 0:
        return None

    center = get_group_center(anchor_group)

    if center is None:
        return None

    start_r = round(center[0])
    start_c = round(center[1])

    h, w = grid_shape(grid)

    exact_candidates = []
    near_line_candidates = []

    max_steps = max(h, w)

    for multiplier in range(1, max_steps + 1):
        distance = step * multiplier

        if direction == "vertical":
            base_points = [
                (start_r + distance, start_c),
                (start_r - distance, start_c),
            ]

            for base_r, base_c in base_points:
                exact_candidates.append((base_r, base_c))

                for side_offset in range(1, 4):
                    near_line_candidates.append((base_r, base_c + side_offset))
                    near_line_candidates.append((base_r, base_c - side_offset))

        elif direction == "horizontal":
            base_points = [
                (start_r, start_c + distance),
                (start_r, start_c - distance),
            ]

            for base_r, base_c in base_points:
                exact_candidates.append((base_r, base_c))

                for side_offset in range(1, 4):
                    near_line_candidates.append((base_r + side_offset, base_c))
                    near_line_candidates.append((base_r - side_offset, base_c))

    if used_generic_rule:
        rule_reason_prefix = (
            f"learned generic {direction} growth rule "
            f"borrowed for {assignment_label}"
        )
    else:
        rule_reason_prefix = (
            f"learned {assignment_label} {direction} growth rule"
        )

    for candidate in exact_candidates:
        r, c = candidate

        if not in_bounds(grid, r, c):
            continue

        if candidate in open_cells:
            return {
                "location": candidate,
                "reason": f"{rule_reason_prefix} exact",
                "anchor_group": anchor_group,
                "pattern_step": step,
                "growth_rule": rule,
                "used_generic_growth_rule": used_generic_rule,
            }

    for candidate in near_line_candidates:
        r, c = candidate

        if not in_bounds(grid, r, c):
            continue

        if candidate in open_cells:
            return {
                "location": candidate,
                "reason": f"{rule_reason_prefix} near-line search",
                "anchor_group": anchor_group,
                "pattern_step": step,
                "growth_rule": rule,
                "used_generic_growth_rule": used_generic_rule,
            }

    return None


def choose_new_marker_location_by_local_template_pattern(
    grid,
    output_template,
    anchor_group,
):
    """
    Fallback placement.

    This uses local marker layout inside the matched output template.
    It is weaker than assignment-growth learning, but still template-based.

    Important:
        This function no longer uses nearest-open-cell guessing.
        If it cannot extend a learned local pattern, it returns None.
    """
    fill_color = output_template.get("fill_color", 0)
    open_cells = set(find_open_interior_cells(grid, fill_color))

    if not open_cells:
        return None

    groups = output_template.get("marker_groups", [])

    point_groups = [
        group for group in groups
        if group.get("cell_count", 0) <= 3 and group.get("bbox") is not None
    ]

    if len(point_groups) >= 2:
        centers = []

        for group in point_groups:
            center = get_group_center(group)

            if center is None:
                continue

            centers.append((round(center[0]), round(center[1])))

        centers = sorted(set(centers))

        by_col = {}

        for r, c in centers:
            by_col.setdefault(c, [])
            by_col[c].append(r)

        for c, rows in by_col.items():
            rows = sorted(rows)

            if len(rows) >= 2:
                diffs = [
                    rows[i + 1] - rows[i]
                    for i in range(len(rows) - 1)
                ]

                step = max(set(diffs), key=diffs.count)

                if step > 0:
                    candidate = (rows[-1] + step, c)

                    if candidate in open_cells:
                        return {
                            "location": candidate,
                            "reason": "local vertical pattern extension forward",
                            "anchor_group": anchor_group,
                            "pattern_step": step,
                            "growth_rule": None,
                        }

                    candidate = (rows[0] - step, c)

                    if candidate in open_cells:
                        return {
                            "location": candidate,
                            "reason": "local vertical pattern extension backward",
                            "anchor_group": anchor_group,
                            "pattern_step": step,
                            "growth_rule": None,
                        }

        by_row = {}

        for r, c in centers:
            by_row.setdefault(r, [])
            by_row[r].append(c)

        for r, cols in by_row.items():
            cols = sorted(cols)

            if len(cols) >= 2:
                diffs = [
                    cols[i + 1] - cols[i]
                    for i in range(len(cols) - 1)
                ]

                step = max(set(diffs), key=diffs.count)

                if step > 0:
                    candidate = (r, cols[-1] + step)

                    if candidate in open_cells:
                        return {
                            "location": candidate,
                            "reason": "local horizontal pattern extension forward",
                            "anchor_group": anchor_group,
                            "pattern_step": step,
                            "growth_rule": None,
                        }

                    candidate = (r, cols[0] - step)

                    if candidate in open_cells:
                        return {
                            "location": candidate,
                            "reason": "local horizontal pattern extension backward",
                            "anchor_group": anchor_group,
                            "pattern_step": step,
                            "growth_rule": None,
                        }

    # No nearest-open-cell guessing.
    #
    # If the learned assignment rule, generic growth rule,
    # or local template pattern cannot place a marker,
    # then we return None and let the action fail.
    return None


def add_marker_for_extra_assignment(
    grid,
    output_template,
    assignment_label,
    growth_rules=None,
    generic_growth_rule=None,
):
    """
    Add one learned marker cell for one extra blob assignment.

    Priority:
        1. assignment-specific learned growth rule
        2. generic task-level growth rule
        3. local template pattern fallback
        4. nearest open-cell fallback

    Returns an action dictionary for the debugger.
    """
    if growth_rules is None:
        growth_rules = {}

    marker_color = get_template_marker_color(output_template)

    anchor_group = choose_anchor_group_for_assignment(
        output_template,
        assignment_label,
    )

    if anchor_group is None:
        return {
            "assignment": assignment_label,
            "success": False,
            "reason": "no anchor group found",
            "placed_at": None,
            "marker_color": marker_color,
            "anchor_group": None,
            "pattern_step": None,
            "growth_rule_used": None,
            "used_generic_growth_rule": False,
        }

    placement = choose_new_marker_location_from_growth_rule(
        grid=grid,
        output_template=output_template,
        anchor_group=anchor_group,
        assignment_label=assignment_label,
        growth_rules=growth_rules,
        generic_growth_rule=generic_growth_rule,
    )

    if placement is None:
        placement = choose_new_marker_location_by_local_template_pattern(
            grid=grid,
            output_template=output_template,
            anchor_group=anchor_group,
        )

    if placement is None:
        return {
            "assignment": assignment_label,
            "success": False,
            "reason": "no valid placement found",
            "placed_at": None,
            "marker_color": marker_color,
            "anchor_group": anchor_group,
            "pattern_step": None,
            "growth_rule_used": None,
            "used_generic_growth_rule": False,
        }

    r, c = placement["location"]
    grid[r][c] = marker_color

    return {
        "assignment": assignment_label,
        "success": True,
        "reason": placement.get("reason"),
        "placed_at": (r, c),
        "marker_color": marker_color,
        "anchor_group": anchor_group,
        "pattern_step": placement.get("pattern_step"),
        "growth_rule_used": placement.get("growth_rule"),
        "used_generic_growth_rule": placement.get(
            "used_generic_growth_rule",
            False,
        ),
    }


# ============================================================
# Adapted prediction
# ============================================================
def create_adapted_prediction_from_match(
    match,
    growth_rules=None,
    generic_growth_rule=None,
    return_actions=False,
):
    """
    Create an adapted prediction from a matched learned template.

    Logic:
        closest train template
        + feature delta
        + assignment-specific marker growth rules
        + generic task-level marker growth rule
    """
    if growth_rules is None:
        growth_rules = {}

    output_template = match["matched_output_template"]
    feature_delta = match.get("feature_delta", {})

    grid = create_prediction_from_template(output_template)

    assignment_delta = feature_delta.get("assignment_delta", {})
    actions = []

    for assignment_label, delta in assignment_delta.items():
        if delta <= 0:
            continue

        for _ in range(delta):
            action = add_marker_for_extra_assignment(
                grid=grid,
                output_template=output_template,
                assignment_label=assignment_label,
                growth_rules=growth_rules,
                generic_growth_rule=generic_growth_rule,
            )
            actions.append(action)

    if return_actions:
        return grid, actions

    return grid


def create_adapted_prediction_from_closest_mapping(
    mapping_rule,
    input_grid,
    return_actions=False,
):
    """
    Adapted prediction:
        closest learned template replay
        + learned feature-delta adjustment
        + learned assignment marker growth rules
        + generic task-level marker growth rule
    """
    match = choose_closest_mapping_for_input(mapping_rule, input_grid)

    if match is None:
        if return_actions:
            return None, []

        return None

    growth_rules = mapping_rule.get("assignment_marker_growth_rules", {})
    generic_growth_rule = mapping_rule.get("generic_marker_growth_rule")

    return create_adapted_prediction_from_match(
        match=match,
        growth_rules=growth_rules,
        generic_growth_rule=generic_growth_rule,
        return_actions=return_actions,
    )