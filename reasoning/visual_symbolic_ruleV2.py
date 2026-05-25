# reasoning/visual_symbolic_ruleV2.py

"""
visual_symbolic_ruleV2

Goal:
    A small, structure-first symbolic learner.

This file intentionally does NOT learn color rules.

It learns:
    - output shape
    - outer frame placement
    - one-ring center markers
    - rule confidence through leave-one-out

Color is only used at final rendering:
    draw_color = first active input color
    fill_color = input background color
"""

try:
    from reasoning.visual_abstraction_mapper import discover_visual_abstractions
except Exception:
    discover_visual_abstractions = None


# ============================================================
# BASIC GRID HELPERS
# ============================================================

def grid_shape(grid):
    return len(grid), len(grid[0]) if grid else 0


def make_grid(height, width, value):
    return [[value for _ in range(width)] for _ in range(height)]


def set_cell(grid, r, c, value):
    h, w = grid_shape(grid)
    if 0 <= r < h and 0 <= c < w:
        grid[r][c] = value


def draw_box(grid, top, left, height, width, value):
    if height <= 0 or width <= 0:
        return

    bottom = top + height - 1
    right = left + width - 1

    for c in range(left, right + 1):
        set_cell(grid, top, c, value)
        set_cell(grid, bottom, c, value)

    for r in range(top, bottom + 1):
        set_cell(grid, r, left, value)
        set_cell(grid, r, right, value)


# ============================================================
# INPUT SCENE FACTS
# ============================================================

def count_colors(grid):
    counts = {}
    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    return counts


def get_background_color(grid):
    counts = count_colors(grid)
    return max(counts, key=counts.get)


def get_active_colors(grid):
    bg = get_background_color(grid)
    counts = count_colors(grid)

    active = [
        color for color in counts
        if color != bg
    ]

    active.sort(key=lambda c: counts[c], reverse=True)
    return active


def get_view(summary, view_name):
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


def extract_scene_facts(input_grid):
    """
    V2 scene extractor.

    This keeps only the structural facts we trust from V1:
        - ring_count
        - blob_count
        - outside_blob_count
        - nested_ring_count
        - ring_blob_counts

    No color learning.
    No output inspection.
    No formula search.
    """
    bg = get_background_color(input_grid)
    active = get_active_colors(input_grid)

    if discover_visual_abstractions is None:
        return {
            "background": bg,
            "active_colors": active,
            "ring_count": 0,
            "blob_count": 0,
            "outside_blob_count": 0,
            "nested_ring_count": 0,
            "ring_blob_counts": (),
            "rings": [],
            "blobs": [],
            "assignments": [],
            "outside_blobs": [],
            "blobs_by_ring": {},
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
            "ring_blob_counts": (),
            "rings": [],
            "blobs": [],
            "assignments": [],
            "outside_blobs": [],
            "blobs_by_ring": {},
        }

    rings = ring_blob.get("rings", [])
    blobs = ring_blob.get("blobs", [])
    assignments = ring_blob.get("assignments", [])

    ring_ids = [ring.get("id") for ring in rings]

    blobs_by_ring = {
        ring_id: []
        for ring_id in ring_ids
    }

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
        ring_blob_counts.append(
            len(blobs_by_ring.get(ring_id, []))
        )

    ring_count = len(rings)
    blob_count = len(blobs)
    outside_blob_count = len(outside_blobs)
    nested_ring_count = max(0, ring_count - 1)

    return {
        "background": bg,
        "active_colors": active,

        "ring_count": ring_count,
        "blob_count": blob_count,
        "outside_blob_count": outside_blob_count,
        "nested_ring_count": nested_ring_count,
        "ring_blob_counts": tuple(ring_blob_counts),

        "rings": rings,
        "blobs": blobs,
        "assignments": assignments,
        "outside_blobs": outside_blobs,
        "blobs_by_ring": blobs_by_ring,
    }


# ============================================================
# STRUCTURE RULES
# ============================================================

def predict_output_shape(scene):
    blob_count = scene["blob_count"]
    nested_ring_count = scene["nested_ring_count"]
    outside_blob_count = scene["outside_blob_count"]

    height = 2 * blob_count + 2 * nested_ring_count + 3
    width = outside_blob_count + 6 * nested_ring_count + 5

    return height, width


def predict_outer_frame(scene, output_height, output_width):
    """
    Outer frame rule.

    Normal scenes:
        frame covers the whole output.

    Nested scenes with outside blobs:
        frame covers the main left structure only.
        The right side is a chamber/extra area, not part of the outer box.
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
    First nested-ring structural rule.

    Learned from V1:
        nested scenes have an inner frame starting at row=2, col=2.
        size depends on nested_ring_count:
            one nested ring -> 5x5 for larger nested scene
            one nested ring with smaller scene -> 3x3 is needed later

    For now, use the common inner box:
        top=2, left=2, height=5, width=5

    This is a probe. If it improves Pair 0 but hurts Pair 3,
    we then learn 5x5 vs 3x3 from scene facts.
    """
    if scene.get("nested_ring_count", 0) <= 0:
        return None

    return {
        "top": 2,
        "left": 2,
        "height": 5,
        "width": 5,
    }


def get_nested_ring_blob_counts(scene):
    counts = scene.get("ring_blob_counts", ())

    if len(counts) < 2:
        return 0, 0

    outer_blob_count = counts[0]
    inner_blob_count = counts[1]

    return outer_blob_count, inner_blob_count


def marker_for_blob(blob, center_r, center_c, side_marker_c, lower_marker_r, output_width):
    assigned_to = blob.get("assigned_to")
    direction = blob.get("direction")

    far_side_c = output_width - 2

    # ------------------------------------------------------------
    # outside blob
    # ------------------------------------------------------------
    if assigned_to == "outside_all":
        return center_r, far_side_c

    # ------------------------------------------------------------
    # inner ring blob
    # assigned_to 1
    # ------------------------------------------------------------
    if assigned_to == 1:
        if direction == "right":
            return center_r, side_marker_c

        if direction == "left":
            return center_r, center_c

        if direction == "above":
            return center_r, center_c

        if direction == "below":
            return lower_marker_r, center_c

        if direction == "center":
            return center_r, center_c

    # ------------------------------------------------------------
    # outer ring blob
    # assigned_to 0
    # ------------------------------------------------------------
    if assigned_to == 0:
        if direction == "right":
            return lower_marker_r, center_c

        if direction == "left":
            return center_r, center_c

        if direction == "above":
            return center_r, side_marker_c

        if direction == "below":
            return lower_marker_r, center_c

        if direction == "center":
            return center_r, center_c

    return center_r, center_c


def predict_nested_markers(scene, output_height, output_width):
    inner = predict_inner_frame(scene)

    if inner is None:
        return []

    markers = []

    outer_blob_count, inner_blob_count = get_nested_ring_blob_counts(scene)

    center_r = inner["top"] + inner["height"] // 2
    center_c = inner["left"] + inner["width"] // 2

    right_c = inner["left"] + inner["width"] - 1
    side_marker_c = right_c + 2

    inner_bottom = inner["top"] + inner["height"] - 1
    lower_marker_r = inner_bottom + 2

    # INNER-RING MARKERS
    if inner_blob_count >= 1:
        markers.append((center_r, center_c))

    if inner_blob_count >= 2:
        if lower_marker_r < output_height:
            markers.append((lower_marker_r, center_c))

    # OUTER-RING / SIDE MARKERS
    if outer_blob_count >= 1:
        if side_marker_c < output_width:
            markers.append((center_r, side_marker_c))

    if outer_blob_count >= 2:
        if lower_marker_r < output_height and side_marker_c < output_width:
            markers.append((lower_marker_r, side_marker_c))

    # OUTSIDE-BLOB EXTENSION MARKERS
    if scene.get("outside_blob_count", 0) > 0:
        side_chamber_c = output_width - 2

        if lower_marker_r < output_height:
            markers.append((lower_marker_r, center_c))

        if side_chamber_c < output_width:
            markers.append((center_r, side_chamber_c))

    unique_markers = []
    seen = set()

    for marker in markers:
        if marker not in seen:
            seen.add(marker)
            unique_markers.append(marker)

    return unique_markers


def predict_one_ring_markers(scene, output_height, output_width):
    """
    Learned from V1:
        one-ring markers are centered.
        row_start = 2
        row_step = 2
        marker_count = blob_count
    """
    if scene["ring_count"] != 1:
        return []

    markers = []
    col = output_width // 2

    for i in range(scene["blob_count"]):
        row = 2 + (2 * i)
        markers.append((row, col))

    return markers


# ============================================================
# RENDER STRUCTURE
# ============================================================

def render_prediction(input_grid, scene):
    output_height, output_width = predict_output_shape(scene)

    fill_color = scene["background"]
    active = scene["active_colors"]

    if active:
        draw_color = active[0]
    else:
        draw_color = fill_color

    out = make_grid(output_height, output_width, fill_color)

    frame = predict_outer_frame(scene, output_height, output_width)

    draw_box(
        out,
        frame["top"],
        frame["left"],
        frame["height"],
        frame["width"],
        draw_color,
    )

    inner = predict_inner_frame(scene)

    if inner is not None:
        draw_box(
            out,
            inner["top"],
            inner["left"],
            inner["height"],
            inner["width"],
            draw_color,
        )

    for r, c in predict_nested_markers(scene, output_height, output_width):
        set_cell(out, r, c, draw_color)

    for r, c in predict_one_ring_markers(scene, output_height, output_width):
        set_cell(out, r, c, draw_color)

    return out


def summarize_blob_geometry(scene):
    blobs = scene.get("blobs", [])
    assignments = scene.get("assignments", [])

    summary = []

    for blob in blobs:
        blob_id = blob.get("id")
        box = blob.get("box", {})

        assigned_to = None

        for assignment in assignments:
            if assignment.get("blob_id") == blob_id:
                assigned_to = assignment.get("assigned_to")
                break

        top = box.get("top")
        bottom = box.get("bottom")
        left = box.get("left")
        right = box.get("right")

        if top is not None and bottom is not None:
            center_r = (top + bottom) / 2
        else:
            center_r = None

        if left is not None and right is not None:
            center_c = (left + right) / 2
        else:
            center_c = None

        summary.append({
            "blob_id": blob_id,
            "assigned_to": assigned_to,
            "box": box,
            "center": (center_r, center_c),
            "size": blob.get("size"),
            "fill_ratio": blob.get("fill_ratio"),
        })

    return summary


def summarize_ring_geometry(scene):
    rings = scene.get("rings", [])

    summary = []

    for ring in rings:
        ring_id = ring.get("id")
        box = ring.get("box", {})

        top = box.get("top")
        bottom = box.get("bottom")
        left = box.get("left")
        right = box.get("right")

        if top is not None and bottom is not None:
            center_r = (top + bottom) / 2
        else:
            center_r = None

        if left is not None and right is not None:
            center_c = (left + right) / 2
        else:
            center_c = None

        summary.append({
            "ring_id": ring_id,
            "box": box,
            "center": (center_r, center_c),
        })

    return summary


def get_ring_geometry_by_id(scene):
    ring_geometry = summarize_ring_geometry(scene)

    by_id = {}

    for ring in ring_geometry:
        by_id[ring["ring_id"]] = ring

    return by_id


def classify_blob_direction(blob_info, ring_info):
    """
    Classify a blob by its position relative to its assigned ring.
    """
    blob_center = blob_info.get("center")
    ring_center = ring_info.get("center")

    if blob_center is None or ring_center is None:
        return "unknown"

    br, bc = blob_center
    rr, rc = ring_center

    if br is None or bc is None or rr is None or rc is None:
        return "unknown"

    dr = br - rr
    dc = bc - rc

    if abs(dr) <= 1.0 and abs(dc) <= 1.0:
        return "center"

    if abs(dr) >= abs(dc):
        if dr < 0:
            return "above"
        return "below"

    if dc < 0:
        return "left"

    return "right"


def summarize_blob_directions(scene):
    blob_geometry = summarize_blob_geometry(scene)
    ring_by_id = get_ring_geometry_by_id(scene)

    directions = []

    for blob in blob_geometry:
        assigned_to = blob.get("assigned_to")

        if assigned_to == "outside_all":
            direction = "outside_all"
        else:
            ring_info = ring_by_id.get(assigned_to)

            if ring_info is None:
                direction = "unknown"
            else:
                direction = classify_blob_direction(blob, ring_info)

        directions.append({
            "blob_id": blob.get("blob_id"),
            "assigned_to": assigned_to,
            "center": blob.get("center"),
            "direction": direction,
        })

    return directions


def explain_visual_symbolic_prediction(input_grid):
    scene = extract_scene_facts(input_grid)

    output_height, output_width = predict_output_shape(scene)
    outer_frame = predict_outer_frame(scene, output_height, output_width)
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
    return {
        "scene": scene,
        "output_shape": (output_height, output_width),
        "outer_frame": outer_frame,
        "inner_frame": inner_frame,
        "one_ring_markers": one_ring_markers,
        "nested_markers": nested_markers,
        "blob_geometry": blob_geometry,
        "ring_geometry": ring_geometry,
        "blob_directions": blob_directions,
        "prediction": prediction,

    }


# ============================================================
# PUBLIC SOLVER ENTRY
# ============================================================

def solve_visual_symbolic_rule_v2(input_grid):
    scene = extract_scene_facts(input_grid)
    return render_prediction(input_grid, scene)


def discover_visual_symbolic_rule_v2_for_task(train_pairs):
    """
    V2 does not build a huge learned object yet.

    First target:
        prove that extracted scene facts + structural renderer
        are small, inspectable, and LOO-testable.
    """
    return {
        "family": "visual_symbolic_rule_v2",
        "version": 2,
        "learns_colors": False,
        "uses_structure_only": True,
        "train_pair_count": len(train_pairs),
    }


def apply_visual_symbolic_rule_v2(rule, input_grid):
    return solve_visual_symbolic_rule_v2(input_grid)