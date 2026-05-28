# ============================================================
# visual_symbolic_ruleV2.py
#
# Purpose:
#   Read the visual scene the way Eric described it:
#
#       1. See rings with blobs.
#       2. Count rings and blobs.
#       3. Find where blobs are:
#           - inside which ring
#           - outside all rings
#           - between outer and inner rings
#           - left/right/above/below/center
#
# Important:
#   This file should NOT hand-code marker placement.
#   It should build a scene tree.
#   Output marker placement must be learned later from train pairs.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

try:
    from reasoning.visual_abstraction_mapper import discover_visual_abstractions
except Exception:
    discover_visual_abstractions = None


# ============================================================
# BASIC GRID HELPERS
# ============================================================

def grid_shape(grid):
    if not grid:
        return 0, 0

    return len(grid), len(grid[0])


def copy_grid(grid):
    return [row[:] for row in grid]


def make_grid(height, width, fill_value):
    return [[fill_value for _ in range(width)] for _ in range(height)]


def in_bounds(grid, r, c):
    h, w = grid_shape(grid)
    return 0 <= r < h and 0 <= c < w


def set_cell(grid, r, c, value):
    if in_bounds(grid, r, c):
        grid[r][c] = value


def color_counts(grid):
    counts = {}

    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1

    return counts


def most_common_color(grid):
    counts = color_counts(grid)

    if not counts:
        return 0

    return max(counts, key=counts.get)


def active_colors(grid):
    background = most_common_color(grid)
    counts = color_counts(grid)

    colors = []

    for color, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if color != background:
            colors.append(color)

    return colors


# ============================================================
# VIEW HELPERS
# ============================================================

def get_view(summary, view_name):
    """
    Safely get a named view from visual_abstraction_mapper output.

    Supports either:
        summary["ring_blob_view"]

    or:
        summary["views"]["ring_blob_view"]

    or:
        summary["views"] = [{"view_type": "ring_blob_view", ...}]
    """
    if summary is None:
        return None

    if isinstance(summary, dict):
        if view_name in summary:
            return summary[view_name]

        views = summary.get("views")

        if isinstance(views, dict):
            return views.get(view_name)

        if isinstance(views, list):
            for view in views:
                if view.get("view_type") == view_name:
                    return view

    return None


# ============================================================
# BOX / GEOMETRY HELPERS
# ============================================================

def normalize_box(box):
    """
    Accepts boxes that may use either:
        top/bottom/left/right

    or:
        min_r/max_r/min_c/max_c

    Returns a normalized dict.
    """
    if box is None:
        return {
            "top": None,
            "bottom": None,
            "left": None,
            "right": None,
            "height": None,
            "width": None,
            "area": None,
        }

    top = box.get("top", box.get("min_r"))
    bottom = box.get("bottom", box.get("max_r"))
    left = box.get("left", box.get("min_c"))
    right = box.get("right", box.get("max_c"))

    if top is None or bottom is None:
        height = box.get("height")
    else:
        height = bottom - top + 1

    if left is None or right is None:
        width = box.get("width")
    else:
        width = right - left + 1

    if height is None or width is None:
        area = box.get("area")
    else:
        area = height * width

    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "height": height,
        "width": width,
        "area": area,
    }


def box_center(box):
    box = normalize_box(box)

    top = box.get("top")
    bottom = box.get("bottom")
    left = box.get("left")
    right = box.get("right")

    if top is None or bottom is None:
        center_r = None
    else:
        center_r = (top + bottom) / 2

    if left is None or right is None:
        center_c = None
    else:
        center_c = (left + right) / 2

    return center_r, center_c


def point_inside_box(point, box):
    r, c = point
    box = normalize_box(box)

    top = box.get("top")
    bottom = box.get("bottom")
    left = box.get("left")
    right = box.get("right")

    if None in [r, c, top, bottom, left, right]:
        return False

    return top <= r <= bottom and left <= c <= right


def box_contains_box(outer_box, inner_box):
    outer_box = normalize_box(outer_box)
    inner_box = normalize_box(inner_box)

    values = [
        outer_box.get("top"),
        outer_box.get("bottom"),
        outer_box.get("left"),
        outer_box.get("right"),
        inner_box.get("top"),
        inner_box.get("bottom"),
        inner_box.get("left"),
        inner_box.get("right"),
    ]

    if any(value is None for value in values):
        return False

    return (
        outer_box["top"] <= inner_box["top"]
        and outer_box["bottom"] >= inner_box["bottom"]
        and outer_box["left"] <= inner_box["left"]
        and outer_box["right"] >= inner_box["right"]
    )


def box_area(box):
    box = normalize_box(box)
    area = box.get("area")

    if area is None:
        return 0

    return area


def classify_relative_direction(source_center, target_center):
    """
    Classify source position relative to target.

    Example:
        source blob center compared to ring center.
    """
    if source_center is None or target_center is None:
        return "unknown"

    sr, sc = source_center
    tr, tc = target_center

    if sr is None or sc is None or tr is None or tc is None:
        return "unknown"

    dr = sr - tr
    dc = sc - tc

    if abs(dr) <= 1.0 and abs(dc) <= 1.0:
        return "center"

    if abs(dr) >= abs(dc):
        if dr < 0:
            return "above"
        return "below"

    if dc < 0:
        return "left"

    return "right"


# ============================================================
# OBJECT NORMALIZATION
# ============================================================

def normalize_ring(raw_ring):
    ring_id = raw_ring.get("id")
    box = normalize_box(raw_ring.get("box", raw_ring.get("bbox")))

    return {
        "id": ring_id,
        "type": "ring",
        "box": box,
        "center": box_center(box),
        "area": box_area(box),
        "raw": raw_ring,
    }


def normalize_blob(raw_blob):
    blob_id = raw_blob.get("id")
    box = normalize_box(raw_blob.get("box", raw_blob.get("bbox")))

    return {
        "id": blob_id,
        "type": "blob",
        "box": box,
        "center": box_center(box),
        "area": box_area(box),
        "size": raw_blob.get("size"),
        "fill_ratio": raw_blob.get("fill_ratio"),
        "raw": raw_blob,
    }


def assignment_for_blob(assignments, blob_id):
    for assignment in assignments:
        if assignment.get("blob_id") == blob_id:
            return assignment.get("assigned_to")

    return None


# ============================================================
# SCENE TREE LEARNING / READING
# ============================================================

def find_parent_ring_for_ring(ring, rings):
    """
    Find the smallest ring that contains this ring.

    This gives us:
        outer ring
            contains inner ring
    """
    candidates = []

    for possible_parent in rings:
        if possible_parent["id"] == ring["id"]:
            continue

        if box_contains_box(possible_parent["box"], ring["box"]):
            candidates.append(possible_parent)

    if not candidates:
        return None

    candidates.sort(key=lambda candidate: candidate["area"])
    return candidates[0]["id"]


def build_ring_relationships(rings):
    relationships = {}

    for ring in rings:
        parent_id = find_parent_ring_for_ring(ring, rings)

        relationships[ring["id"]] = {
            "ring_id": ring["id"],
            "parent_ring": parent_id,
            "child_rings": [],
            "contained_blobs": [],
        }

    for ring_id, info in relationships.items():
        parent_id = info["parent_ring"]

        if parent_id in relationships:
            relationships[parent_id]["child_rings"].append(ring_id)

    return relationships


def build_blob_relationships(blobs, rings, assignments):
    ring_by_id = {}

    for ring in rings:
        ring_by_id[ring["id"]] = ring

    blob_infos = []

    for blob in blobs:
        assigned_to = assignment_for_blob(assignments, blob["id"])

        if assigned_to == "outside_all":
            container_ring = None
            direction_in_container = "outside_all"
        else:
            container_ring = assigned_to
            ring = ring_by_id.get(container_ring)

            if ring is None:
                direction_in_container = "unknown"
            else:
                direction_in_container = classify_relative_direction(
                    blob["center"],
                    ring["center"],
                )

        blob_infos.append({
            "blob_id": blob["id"],
            "container_ring": container_ring,
            "assigned_to": assigned_to,
            "center": blob["center"],
            "box": blob["box"],
            "direction_in_container": direction_in_container,
            "outside_all": assigned_to == "outside_all",
        })

    return blob_infos


def attach_blobs_to_rings(ring_relationships, blob_relationships):
    for blob in blob_relationships:
        container_ring = blob["container_ring"]

        if container_ring in ring_relationships:
            ring_relationships[container_ring]["contained_blobs"].append(
                blob["blob_id"]
            )


def classify_ring_roles(rings, ring_relationships):
    """
    Give rings structural roles.

    This is not task-output hardcoding.
    This is scene understanding.

    Examples:
        parent_ring is None       -> outer
        has parent and no child   -> inner
        has parent and has child  -> middle
    """
    roles = {}

    for ring in rings:
        ring_id = ring["id"]
        info = ring_relationships.get(ring_id, {})

        parent = info.get("parent_ring")
        children = info.get("child_rings", [])

        if parent is None and children:
            role = "outer"
        elif parent is None and not children:
            role = "single"
        elif parent is not None and children:
            role = "middle"
        else:
            role = "inner"

        roles[ring_id] = role

    return roles


def build_scene_tree(background, active, raw_rings, raw_blobs, assignments):
    rings = [normalize_ring(ring) for ring in raw_rings]
    blobs = [normalize_blob(blob) for blob in raw_blobs]

    rings.sort(key=lambda ring: ring["area"], reverse=True)
    blobs.sort(key=lambda blob: blob["id"])

    ring_relationships = build_ring_relationships(rings)
    blob_relationships = build_blob_relationships(blobs, rings, assignments)

    attach_blobs_to_rings(ring_relationships, blob_relationships)

    ring_roles = classify_ring_roles(rings, ring_relationships)

    ring_nodes = []

    for ring in rings:
        ring_id = ring["id"]
        rel = ring_relationships.get(ring_id, {})

        ring_nodes.append({
            "id": ring_id,
            "node_id": f"ring_{ring_id}",
            "type": "ring",
            "role": ring_roles.get(ring_id, "unknown"),
            "box": ring["box"],
            "center": ring["center"],
            "parent_ring": rel.get("parent_ring"),
            "child_rings": rel.get("child_rings", []),
            "contained_blobs": rel.get("contained_blobs", []),
        })

    blob_nodes = []

    for blob in blobs:
        blob_info = None

        for candidate in blob_relationships:
            if candidate["blob_id"] == blob["id"]:
                blob_info = candidate
                break

        if blob_info is None:
            blob_info = {
                "blob_id": blob["id"],
                "container_ring": None,
                "assigned_to": None,
                "center": blob["center"],
                "direction_in_container": "unknown",
                "outside_all": False,
            }

        blob_nodes.append({
            "id": blob["id"],
            "node_id": f"blob_{blob['id']}",
            "type": "blob",
            "box": blob["box"],
            "center": blob["center"],
            "container_ring": blob_info["container_ring"],
            "assigned_to": blob_info["assigned_to"],
            "direction_in_container": blob_info["direction_in_container"],
            "outside_all": blob_info["outside_all"],
            "size": blob.get("size"),
            "fill_ratio": blob.get("fill_ratio"),
        })

    return {
        "background": background,
        "active_colors": active,
        "rings": ring_nodes,
        "blobs": blob_nodes,
        "assignments": assignments,
    }


def get_outer_rings(scene_tree):
    result = []

    for ring in scene_tree["rings"]:
        if ring["role"] == "outer":
            result.append(ring)

    return result


def get_inner_rings(scene_tree):
    result = []

    for ring in scene_tree["rings"]:
        if ring["role"] == "inner":
            result.append(ring)

    return result


def get_single_rings(scene_tree):
    result = []

    for ring in scene_tree["rings"]:
        if ring["role"] == "single":
            result.append(ring)

    return result


def get_blobs_inside_ring(scene_tree, ring_id):
    result = []

    for blob in scene_tree["blobs"]:
        if blob["container_ring"] == ring_id:
            result.append(blob)

    return result


def get_outside_blobs(scene_tree):
    result = []

    for blob in scene_tree["blobs"]:
        if blob["outside_all"]:
            result.append(blob)

    return result


def get_ring_by_id(scene_tree, ring_id):
    for ring in scene_tree["rings"]:
        if ring["id"] == ring_id:
            return ring

    return None


# ============================================================
# HUMAN-STYLE SCENE SUMMARY
# ============================================================

def summarize_scene_as_eric_sees_it(scene_tree):
    """
    This is the important debug view.

    It speaks in the order Eric described:
        rings
        blobs
        where blobs are in rings
    """
    lines = []

    rings = scene_tree["rings"]
    blobs = scene_tree["blobs"]

    lines.append(f"rings: {len(rings)}")
    lines.append(f"blobs: {len(blobs)}")

    for ring in rings:
        ring_id = ring["id"]
        role = ring["role"]
        child_rings = ring["child_rings"]
        contained_blobs = ring["contained_blobs"]

        lines.append(
            f"ring_{ring_id}: role={role}, "
            f"child_rings={child_rings}, "
            f"contained_blobs={contained_blobs}"
        )

        for blob in get_blobs_inside_ring(scene_tree, ring_id):
            lines.append(
                f"  blob_{blob['id']} inside ring_{ring_id}, "
                f"direction={blob['direction_in_container']}"
            )

    outside_blobs = get_outside_blobs(scene_tree)

    for blob in outside_blobs:
        lines.append(
            f"blob_{blob['id']} outside_all"
        )

    return lines


def build_flat_scene_facts(scene_tree):
    """
    Keep old-style facts for the runner and debugging,
    but these facts are now derived from the scene tree.
    """
    rings = scene_tree["rings"]
    blobs = scene_tree["blobs"]
    outside_blobs = get_outside_blobs(scene_tree)

    ring_blob_counts = []

    for ring in rings:
        ring_blob_counts.append(len(get_blobs_inside_ring(scene_tree, ring["id"])))

    nested_ring_count = 0

    for ring in rings:
        if ring["parent_ring"] is not None:
            nested_ring_count += 1

    return {
        "background": scene_tree["background"],
        "active_colors": scene_tree["active_colors"],
        "ring_count": len(rings),
        "blob_count": len(blobs),
        "outside_blob_count": len(outside_blobs),
        "nested_ring_count": nested_ring_count,
        "ring_blob_counts": tuple(ring_blob_counts),
        "outside_blobs": [blob["id"] for blob in outside_blobs],
        "scene_tree": scene_tree,
    }


# ============================================================
# SCENE EXTRACTION
# ============================================================

def extract_scene_facts(input_grid):
    """
    Main scene reader.

    This function should read the input, not solve the output.
    """
    bg = most_common_color(input_grid)
    active = active_colors(input_grid)

    if discover_visual_abstractions is None:
        return {
            "background": bg,
            "active_colors": active,
            "ring_count": 0,
            "blob_count": 0,
            "outside_blob_count": 0,
            "nested_ring_count": 0,
            "ring_blob_counts": tuple(),
            "outside_blobs": [],
            "scene_tree": {
                "background": bg,
                "active_colors": active,
                "rings": [],
                "blobs": [],
                "assignments": [],
            },
        }

    summary = discover_visual_abstractions(input_grid)
    ring_blob = get_view(summary, "ring_blob_view")

    if ring_blob is None:
        return {
            "background": bg,
            "active_colors": active,
            "ring_count": 0,
            "blob_count": 0,
            "outside_blob_count": 0,
            "nested_ring_count": 0,
            "ring_blob_counts": tuple(),
            "outside_blobs": [],
            "scene_tree": {
                "background": bg,
                "active_colors": active,
                "rings": [],
                "blobs": [],
                "assignments": [],
            },
        }

    raw_rings = ring_blob.get("rings", [])
    raw_blobs = ring_blob.get("blobs", [])
    assignments = ring_blob.get("assignments", [])

    scene_tree = build_scene_tree(
        bg,
        active,
        raw_rings,
        raw_blobs,
        assignments,
    )

    return build_flat_scene_facts(scene_tree)


# ============================================================
# GEOMETRY SUMMARIES
# ============================================================

def summarize_ring_geometry(scene):
    scene_tree = scene.get("scene_tree", {})
    rings = scene_tree.get("rings", [])

    summary = []

    for ring in rings:
        summary.append({
            "ring_id": ring["id"],
            "role": ring["role"],
            "box": ring["box"],
            "center": ring["center"],
            "parent_ring": ring["parent_ring"],
            "child_rings": ring["child_rings"],
            "contained_blobs": ring["contained_blobs"],
        })

    return summary


def summarize_blob_geometry(scene):
    scene_tree = scene.get("scene_tree", {})
    blobs = scene_tree.get("blobs", [])

    summary = []

    for blob in blobs:
        summary.append({
            "blob_id": blob["id"],
            "assigned_to": blob["assigned_to"],
            "container_ring": blob["container_ring"],
            "box": blob["box"],
            "center": blob["center"],
            "direction_in_container": blob["direction_in_container"],
            "outside_all": blob["outside_all"],
            "size": blob.get("size"),
            "fill_ratio": blob.get("fill_ratio"),
        })

    return summary


def summarize_blob_directions(scene):
    scene_tree = scene.get("scene_tree", {})
    blobs = scene_tree.get("blobs", [])

    directions = []

    for blob in blobs:
        directions.append({
            "blob_id": blob["id"],
            "assigned_to": blob["assigned_to"],
            "container_ring": blob["container_ring"],
            "center": blob["center"],
            "direction": blob["direction_in_container"],
            "outside_all": blob["outside_all"],
        })

    return directions


def summarize_scene_tree(scene):
    scene_tree = scene.get("scene_tree", {})
    return summarize_scene_as_eric_sees_it(scene_tree)


# ============================================================
# OUTPUT SHAPE / FRAME VIEW
# ============================================================

def predict_output_shape(scene):
    """
    Temporary frame-view shape.

    This is not the final learned output rule.
    It exists so the runner can still display a prediction grid.

    The real output construction should move to a learner later.
    """
    blob_count = scene["blob_count"]
    nested_ring_count = scene["nested_ring_count"]
    outside_blob_count = scene["outside_blob_count"]

    height = 2 * blob_count + 2 * nested_ring_count + 3
    width = outside_blob_count + 6 * nested_ring_count + 5

    return height, width


def predict_outer_frame(scene, output_height, output_width):
    """
    Temporary frame view.

    Kept only for visualization/debug.
    Not marker solving.
    """
    width = output_width

    if (
        scene.get("nested_ring_count", 0) > 0
        and scene.get("outside_blob_count", 0) > 0
    ):
        width = output_width - 3

    return {
        "top": 0,
        "left": 0,
        "height": output_height,
        "width": width,
    }


def predict_inner_frame(scene):
    """
    Temporary frame view.

    Kept only for visualization/debug.
    """
    if scene.get("nested_ring_count", 0) <= 0:
        return None

    return {
        "top": 2,
        "left": 2,
        "height": 5,
        "width": 5,
    }


def draw_box(grid, top, left, height, width, color):
    bottom = top + height - 1
    right = left + width - 1

    for c in range(left, right + 1):
        set_cell(grid, top, c, color)
        set_cell(grid, bottom, c, color)

    for r in range(top, bottom + 1):
        set_cell(grid, r, left, color)
        set_cell(grid, r, right, color)


# ============================================================
# MARKERS REMOVED
# ============================================================

def predict_one_ring_markers(scene, output_height, output_width):
    """
    Marker prediction removed.

    V2 should not guess output markers by hand.
    Marker placement must be learned from train pairs.
    """
    return []


def predict_nested_markers(scene, output_height, output_width):
    """
    Marker prediction removed.

    V2 reads the scene only.
    Marker placement must be learned from train pairs.
    """
    return []


# ============================================================
# MARKER LEARNING DEBUG VIEW
# ============================================================

def summarize_marker_learning_case(scene, output_height, output_width):
    """
    This does NOT solve markers.

    It only shows available symbolic scene facts that a later learner
    can use.
    """
    scene_tree = scene.get("scene_tree", {})

    return {
        "mode": "scene_reader_only",
        "scene_summary": summarize_scene_as_eric_sees_it(scene_tree),
        "blob_directions": summarize_blob_directions(scene),
        "predicted_markers": [],
        "active_slots": [],
    }


# ============================================================
# RENDERING
# ============================================================

def render_prediction(input_grid, scene):
    """
    Render a frame-only visualization.

    This is intentionally incomplete because marker logic was removed.
    """
    output_height, output_width = predict_output_shape(scene)

    fill_color = scene["background"]
    active = scene["active_colors"]

    if active:
        draw_color = active[0]
    else:
        draw_color = fill_color

    out = make_grid(output_height, output_width, fill_color)

    outer_frame = predict_outer_frame(scene, output_height, output_width)

    draw_box(
        out,
        outer_frame["top"],
        outer_frame["left"],
        outer_frame["height"],
        outer_frame["width"],
        draw_color,
    )

    inner_frame = predict_inner_frame(scene)

    if inner_frame is not None:
        draw_box(
            out,
            inner_frame["top"],
            inner_frame["left"],
            inner_frame["height"],
            inner_frame["width"],
            draw_color,
        )

    return out


# ============================================================
# MAIN PUBLIC EXPLANATION FUNCTION
# ============================================================

def explain_visual_symbolic_prediction(input_grid):
    scene = extract_scene_facts(input_grid)

    output_height, output_width = predict_output_shape(scene)

    outer_frame = predict_outer_frame(
        scene,
        output_height,
        output_width,
    )

    inner_frame = predict_inner_frame(scene)

    one_ring_markers = predict_one_ring_markers(
        scene,
        output_height,
        output_width,
    )

    nested_markers = predict_nested_markers(
        scene,
        output_height,
        output_width,
    )

    prediction = render_prediction(input_grid, scene)

    blob_geometry = summarize_blob_geometry(scene)
    ring_geometry = summarize_ring_geometry(scene)
    blob_directions = summarize_blob_directions(scene)
    scene_tree_summary = summarize_scene_tree(scene)

    marker_learning_case = summarize_marker_learning_case(
        scene,
        output_height,
        output_width,
    )

    return {
        "scene": scene,
        "scene_tree": scene.get("scene_tree"),
        "scene_tree_summary": scene_tree_summary,

        "output_shape": (output_height, output_width),
        "outer_frame": outer_frame,
        "inner_frame": inner_frame,

        "one_ring_markers": one_ring_markers,
        "nested_markers": nested_markers,

        "blob_geometry": blob_geometry,
        "ring_geometry": ring_geometry,
        "blob_directions": blob_directions,

        "marker_learning_case": marker_learning_case,
        "prediction": prediction,
    }


# ============================================================
# RULE DISCOVERY WRAPPER
# ============================================================

def discover_visual_symbolic_rule_v2_for_task(train_pairs):
    """
    Discover scene structure from train pairs.

    This does not learn marker placement yet.
    """
    examples = []

    for idx, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        output_grid = pair.get("output")

        explanation = explain_visual_symbolic_prediction(input_grid)
        scene = explanation["scene"]

        example = {
            "pair_index": idx,
            "scene": scene,
            "scene_tree_summary": explanation["scene_tree_summary"],
            "output_shape": explanation["output_shape"],
        }

        if output_grid is not None:
            example["expected_output_shape"] = grid_shape(output_grid)

        examples.append(example)

    return {
        "family": "visual_symbolic_rule_v2",
        "mode": "scene_reader_only",
        "train_pair_count": len(train_pairs),
        "examples": examples,
    }


def apply_visual_symbolic_rule_v2(rule, input_grid):
    """
    Apply scene reader and frame-only render.

    Marker placement intentionally removed.
    """
    explanation = explain_visual_symbolic_prediction(input_grid)
    return explanation["prediction"]


def solve_visual_symbolic_rule_v2(train_pairs, test_input):
    """
    Convenience solver wrapper.
    """
    rule = discover_visual_symbolic_rule_v2_for_task(train_pairs)
    return apply_visual_symbolic_rule_v2(rule, test_input)


# ============================================================
# OPTIONAL ALIASES FOR ROUTER COMPATIBILITY
# ============================================================

def discover_rule_for_task(train_pairs):
    return discover_visual_symbolic_rule_v2_for_task(train_pairs)


def apply_rule(rule, input_grid):
    return apply_visual_symbolic_rule_v2(rule, input_grid)