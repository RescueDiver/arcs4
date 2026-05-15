# reasoning/ring_blob_scene.py
"""
Ring Blob Scene Learner

Goal:
    Teach the solver to see the input the way a human sees it:

        - rings
        - blobs
        - which blobs are inside which ring
        - whether blobs are single, side-by-side, stacked, or clustered

This file does NOT solve the ARC task.
It only builds a reusable visual scene description.

This is not task hardcoding.
It uses general visual rules:

    1. Rings are enclosed background regions.
    2. Blobs are smaller foreground components.
    3. A blob belongs to the ring whose interior contains the blob center.
    4. Blob layout is learned from blob center positions.
"""

from collections import deque

from reasoning.ring_detector import (
    grid_shape,
    in_bounds,
    neighbors4,
    get_background_color,
    get_foreground_colors,
    connected_components_for_color,
    bbox_for_cells,
    center_for_cells,
    find_ring_components,
)


# ============================================================
# Basic geometry helpers
# ============================================================

def bbox_area(bbox):
    if bbox is None:
        return 0

    r1, c1, r2, c2 = bbox

    return (r2 - r1 + 1) * (c2 - c1 + 1)


def bbox_height(bbox):
    if bbox is None:
        return 0

    r1, c1, r2, c2 = bbox

    return r2 - r1 + 1


def bbox_width(bbox):
    if bbox is None:
        return 0

    r1, c1, r2, c2 = bbox

    return c2 - c1 + 1


def point_inside_bbox(point, bbox):
    if point is None or bbox is None:
        return False

    r, c = point
    r1, c1, r2, c2 = bbox

    return r1 <= r <= r2 and c1 <= c <= c2


def point_inside_cell_set(point, cells):
    """
    Check whether rounded point lands inside a cell set.
    """
    if point is None:
        return False

    r, c = point
    rounded_point = (round(r), round(c))

    return rounded_point in cells


def distance_sq(a, b):
    if a is None or b is None:
        return 10 ** 12

    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


# ============================================================
# Foreground component / blob discovery
# ============================================================

def get_all_foreground_components(grid, background_color=None):
    """
    Find all connected foreground components for every non-background color.
    """
    if background_color is None:
        background_color = get_background_color(grid)

    components = []

    for color in get_foreground_colors(grid, background_color):
        color_components = connected_components_for_color(grid, color)

        for cells in color_components:
            bbox = bbox_for_cells(cells)
            center = center_for_cells(cells)

            component = {
                "color": color,
                "cells": sorted(cells),
                "cell_count": len(cells),
                "bbox": bbox,
                "height": bbox_height(bbox),
                "width": bbox_width(bbox),
                "bbox_area": bbox_area(bbox),
                "center": center,
            }

            components.append(component)

    components.sort(
        key=lambda item: (
            item["color"],
            item["bbox"][0],
            item["bbox"][1],
            item["cell_count"],
        )
    )

    return components


def component_overlaps_ring_wall(component, rings):
    """
    A big foreground component may be part of a ring wall.
    We do not want to call those blobs.

    We check whether any component cell is part of any ring wall-touching set.
    """
    component_cells = set(component.get("cells", []))

    for ring in rings:
        wall_cells = set(ring.get("wall_touching_cells", []))

        if component_cells & wall_cells:
            return True

    return False


def component_inside_any_ring_interior(component, rings):
    """
    A blob should have its center inside a ring interior bbox or exact interior set.

    We allow bbox containment because blobs often occupy foreground cells,
    not the background cells themselves.
    """
    center = component.get("center")

    for ring in rings:
        interior_cells = set(ring.get("interior_cells", []))
        interior_bbox = ring.get("interior_bbox")

        if point_inside_cell_set(center, interior_cells):
            return True

        if point_inside_bbox(center, interior_bbox):
            return True

    return False


def looks_like_blob_component(component, rings):
    """
    Decide whether a foreground component is a blob.

    A blob should be:
        - compact
        - NOT the ring wall itself

    Important:
        Blobs may be inside a ring OR outside all rings.

    So we do NOT require:
        component_inside_any_ring_interior(component, rings)

    because that would miss outside blobs.
    """
    cell_count = component.get("cell_count", 0)
    height = component.get("height", 0)
    width = component.get("width", 0)
    bbox = component.get("bbox")

    if cell_count <= 0:
        return False

    # ------------------------------------------------------------
    # Reject ring-wall-like components.
    # ------------------------------------------------------------
    for ring in rings:
        wall_bbox = ring.get("wall_bbox")
        wall_cell_count = ring.get("wall_touching_cell_count", 0)

        if bbox == wall_bbox:
            return False

        if wall_cell_count > 0:
            ratio = cell_count / wall_cell_count

            if 0.75 <= ratio <= 1.25:
                component_area = bbox_area(bbox)
                wall_area = bbox_area(wall_bbox)

                if wall_area > 0:
                    area_ratio = component_area / wall_area

                    if 0.75 <= area_ratio <= 1.25:
                        return False

    # ------------------------------------------------------------
    # Normal blob filters.
    # ------------------------------------------------------------
    if cell_count > 40:
        return False

    if height > 8 or width > 8:
        return False

    return True


def find_blob_components(grid, rings, background_color=None):
    """
    Find likely blob components.
    """
    if background_color is None:
        background_color = get_background_color(grid)

    all_components = get_all_foreground_components(
        grid,
        background_color,
    )

    blobs = []

    for component in all_components:
        if not looks_like_blob_component(component, rings):
            continue

        blob = dict(component)
        blob["blob_index"] = len(blobs)
        blobs.append(blob)

    return blobs


# ============================================================
# Blob assignment to rings
# ============================================================

def assign_blob_to_ring(blob, rings):
    """
    Assign a blob to the smallest ring whose interior contains the blob center.

    If no ring contains it, return None.
    That means the blob is outside all rings.
    """
    center = blob.get("center")
    candidates = []

    for ring in rings:
        interior_cells = set(ring.get("interior_cells", []))
        interior_bbox = ring.get("interior_bbox")

        contains = False

        if point_inside_cell_set(center, interior_cells):
            contains = True

        if point_inside_bbox(center, interior_bbox):
            contains = True

        if not contains:
            continue

        candidates.append(ring)

    if not candidates:
        return None

    # Smallest containing ring wins.
    # This handles nested rings correctly.
    candidates.sort(
        key=lambda ring: (
            ring.get("interior_cell_count", 10 ** 9),
            bbox_area(ring.get("interior_bbox")),
        )
    )

    return candidates[0]


def group_blobs_by_ring(blobs, rings):
    """
    Group blobs by assigned ring.
    """
    groups = {}

    for ring in rings:
        ring_index = ring["ring_index"]
        groups[ring_index] = []

    outside_blobs = []

    for blob in blobs:
        assigned_ring = assign_blob_to_ring(blob, rings)

        if assigned_ring is None:
            outside_blobs.append(blob)
            blob["assigned_ring_index"] = None
            blob["assigned_ring_label"] = "outside"
        else:
            ring_index = assigned_ring["ring_index"]
            groups[ring_index].append(blob)
            blob["assigned_ring_index"] = ring_index
            blob["assigned_ring_label"] = f"ring_{ring_index}"

    return groups, outside_blobs


# ============================================================
# Blob layout description
# ============================================================

def describe_blob_layout(blobs):
    """
    Learn the layout of blobs from their centers.

    General rules:
        - 0 blobs: empty
        - 1 blob : single
        - 2 blobs:
            similar row + different cols -> side_by_side_horizontal
            similar col + different rows -> stacked_vertical
        - 3+ blobs:
            mostly same row -> row_group
            mostly same col -> column_group
            otherwise       -> cluster
    """
    if not blobs:
        return "empty"

    if len(blobs) == 1:
        return "single"

    centers = [
        blob.get("center")
        for blob in blobs
        if blob.get("center") is not None
    ]

    if len(centers) != len(blobs):
        return "unknown"

    rows = [center[0] for center in centers]
    cols = [center[1] for center in centers]

    row_span = max(rows) - min(rows)
    col_span = max(cols) - min(cols)

    if len(blobs) == 2:
        # Similar rows, separated columns.
        if row_span <= 2 and col_span > row_span:
            return "side_by_side_horizontal"

        # Similar columns, separated rows.
        if col_span <= 2 and row_span > col_span:
            return "stacked_vertical"

        return "diagonal_or_offset_pair"

    # 3 or more blobs.
    if row_span <= 2 and col_span > row_span:
        return "row_group"

    if col_span <= 2 and row_span > col_span:
        return "column_group"

    return "cluster"


# ============================================================
# Full scene learning
# ============================================================

def learn_ring_blob_scene(grid):
    """
    Learn a human-style scene description from one input grid.

    Output example:
        {
            "background_color": 7,
            "ring_count": 2,
            "blob_count": 3,
            "rings": [
                {
                    "ring_index": 0,
                    "color": 9,
                    "interior_bbox": (...),
                    "blob_count": 1,
                    "blob_layout": "single",
                    "blobs": [...]
                },
                {
                    "ring_index": 1,
                    "color": 9,
                    "interior_bbox": (...),
                    "blob_count": 2,
                    "blob_layout": "side_by_side_horizontal",
                    "blobs": [...]
                }
            ],
            "outside_blob_count": 0,
            "outside_blobs": [...]
        }
    """
    background_color = get_background_color(grid)

    rings = find_ring_components(
        grid,
        background_color,
    )

    blobs = find_blob_components(
        grid,
        rings,
        background_color,
    )

    grouped_blobs, outside_blobs = group_blobs_by_ring(
        blobs,
        rings,
    )

    ring_descriptions = []

    for ring in rings:
        ring_index = ring["ring_index"]
        ring_blobs = grouped_blobs.get(ring_index, [])
        blob_layout = describe_blob_layout(ring_blobs)

        ring_description = {
            "ring_index": ring_index,
            "ring_label": f"ring_{ring_index}",
            "color": ring.get("color"),
            "interior_cell_count": ring.get("interior_cell_count"),
            "interior_bbox": ring.get("interior_bbox"),
            "wall_bbox": ring.get("wall_bbox"),
            "center": ring.get("center"),
            "blob_count": len(ring_blobs),
            "blob_layout": blob_layout,
            "blobs": ring_blobs,
        }

        ring_descriptions.append(ring_description)

    scene = {
        "background_color": background_color,
        "ring_count": len(rings),
        "blob_count": len(blobs),
        "rings": ring_descriptions,
        "outside_blob_count": len(outside_blobs),
        "outside_blobs": outside_blobs,
    }

    return scene


def print_ring_blob_scene_summary(grid):
    """
    Debug helper for humans.

    This should eventually print what Eric sees:
        2 rings
        3 blobs
        ring_0 has 1 blob
        ring_1 has 2 blobs side by side
    """
    scene = learn_ring_blob_scene(grid)

    print()
    print("RING/BLOB SCENE SUMMARY")
    print("-" * 60)
    print(f"background color  : {scene.get('background_color')}")
    print(f"ring count        : {scene.get('ring_count')}")
    print(f"blob count        : {scene.get('blob_count')}")
    print(f"outside blob count: {scene.get('outside_blob_count')}")

    for ring in scene.get("rings", []):
        print()
        print(f"{ring.get('ring_label')}")
        print(f"  color          : {ring.get('color')}")
        print(f"  interior bbox  : {ring.get('interior_bbox')}")
        print(f"  wall bbox      : {ring.get('wall_bbox')}")
        print(f"  center         : {ring.get('center')}")
        print(f"  blobs inside   : {ring.get('blob_count')}")
        print(f"  blob layout    : {ring.get('blob_layout')}")

        for blob in ring.get("blobs", []):
            print(
                "    blob "
                f"{blob.get('blob_index')}: "
                f"color={blob.get('color')}, "
                f"cells={blob.get('cell_count')}, "
                f"bbox={blob.get('bbox')}, "
                f"center={blob.get('center')}"
            )

    if scene.get("outside_blobs"):
        print()
        print("outside blobs")

        for blob in scene.get("outside_blobs", []):
            print(
                "    blob "
                f"{blob.get('blob_index')}: "
                f"color={blob.get('color')}, "
                f"cells={blob.get('cell_count')}, "
                f"bbox={blob.get('bbox')}, "
                f"center={blob.get('center')}"
            )

    return scene