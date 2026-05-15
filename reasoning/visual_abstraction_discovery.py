# reasoning/visual_abstraction_discovery.py

from collections import Counter, deque


# ============================================================
# VISUAL ABSTRACTION DISCOVERY
# ============================================================
#
# This file does NOT solve ARC tasks directly.
#
# Its job is to create multiple possible ways to "see" the input.
#
# Later, learned_region_rule.py or task_router.py can score these views
# against the train outputs and learn which abstraction works best.
#
# Important:
#   We do not want to force:
#       "this is rings and blobs"
#
#   We want to generate possible interpretations:
#       - largest component + extras
#       - all components
#       - possible rings + blobs
#       - foreground box
#
#   Then the solver learns which one explains the train pairs.
#
# ============================================================


# ============================================================
# BASIC GRID HELPERS
# ============================================================

def grid_shape(grid):
    if grid is None:
        return 0, 0

    h = len(grid)
    w = len(grid[0]) if h else 0

    return h, w


def color_counts(grid):
    if grid is None:
        return Counter()

    return Counter(v for row in grid for v in row)


def get_background_color(grid):
    counts = color_counts(grid)

    if not counts:
        return 0

    return counts.most_common(1)[0][0]


def get_active_colors(grid):
    bg = get_background_color(grid)
    counts = color_counts(grid)

    return [
        color
        for color, count in counts.most_common()
        if color != bg
    ]


def in_bounds(grid, r, c):
    h, w = grid_shape(grid)
    return 0 <= r < h and 0 <= c < w


# ============================================================
# BOX HELPERS
# ============================================================

def make_box_from_cells(cells):
    """
    cells may be:
        [(r, c), ...]
    or:
        [(r, c, color), ...]
    """
    if not cells:
        return None

    rows = [cell[0] for cell in cells]
    cols = [cell[1] for cell in cells]

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


def box_contains_point(box, r, c):
    if box is None:
        return False

    return (
        box["top"] <= r <= box["bottom"]
        and box["left"] <= c <= box["right"]
    )


def box_contains_box(outer_box, inner_box):
    if outer_box is None or inner_box is None:
        return False

    return (
        outer_box["top"] <= inner_box["top"]
        and outer_box["bottom"] >= inner_box["bottom"]
        and outer_box["left"] <= inner_box["left"]
        and outer_box["right"] >= inner_box["right"]
    )


def box_center(box):
    if box is None:
        return None

    return (
        (box["top"] + box["bottom"]) / 2,
        (box["left"] + box["right"]) / 2,
    )


def get_foreground_box(grid):
    if grid is None:
        return None

    bg = get_background_color(grid)
    h, w = grid_shape(grid)

    cells = []

    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg:
                cells.append((r, c))

    return make_box_from_cells(cells)


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

def find_components(grid):
    """
    Find 4-connected non-background components.

    Returns largest first.
    """
    if grid is None:
        return []

    bg = get_background_color(grid)
    h, w = grid_shape(grid)

    seen = set()
    components = []

    for sr in range(h):
        for sc in range(w):
            if (sr, sc) in seen:
                continue

            if grid[sr][sc] == bg:
                continue

            q = deque([(sr, sc)])
            seen.add((sr, sc))

            cells = []
            colors = set()

            while q:
                r, c = q.popleft()
                color = grid[r][c]

                cells.append((r, c, color))
                colors.add(color)

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = r + dr
                    nc = c + dc

                    if not in_bounds(grid, nr, nc):
                        continue

                    if (nr, nc) in seen:
                        continue

                    if grid[nr][nc] == bg:
                        continue

                    seen.add((nr, nc))
                    q.append((nr, nc))

            box = make_box_from_cells(cells)

            if box is None:
                continue

            box_area = max(1, box["area"])
            fill_ratio = len(cells) / box_area

            components.append({
                "id": len(components),
                "cells": cells,
                "colors": tuple(sorted(colors)),
                "size": len(cells),
                "box": box,
                "fill_ratio": fill_ratio,
            })

    components.sort(key=lambda comp: comp["size"], reverse=True)

    # Reassign ids after sorting so id 0 is biggest component.
    for idx, comp in enumerate(components):
        comp["id"] = idx

    return components


# ============================================================
# COMPONENT CLASSIFICATION
# ============================================================

def looks_like_ring(component):
    """
    Soft guess, not a hard rule.

    A ring/outline usually:
        - has a reasonably large box
        - has low fill ratio
        - has empty space inside its box
    """
    box = component.get("box")

    if box is None:
        return False

    h = box["height"]
    w = box["width"]
    fill_ratio = component.get("fill_ratio", 1.0)

    if h < 5 or w < 5:
        return False

    if fill_ratio > 0.55:
        return False

    return True


def looks_like_blob(component):
    """
    Soft guess, not a hard rule.

    A blob usually:
        - is smaller
        - is more compact
        - has higher fill ratio
    """
    box = component.get("box")

    if box is None:
        return False

    h = box["height"]
    w = box["width"]
    fill_ratio = component.get("fill_ratio", 0.0)

    if h <= 4 and w <= 4:
        return True

    if fill_ratio >= 0.55:
        return True

    return False


def classify_components_soft(components):
    """
    Give every component a soft label.

    This is not the final answer.
    This only helps build candidate abstractions.
    """
    out = []

    for comp in components:
        label = "unknown"

        if looks_like_ring(comp):
            label = "possible_ring"
        elif looks_like_blob(comp):
            label = "possible_blob"

        item = dict(comp)
        item["soft_label"] = label
        out.append(item)

    return out


# ============================================================
# ABSTRACTION VIEW BUILDERS
# ============================================================

def build_largest_component_view(grid, components):
    """
    View 1:
        largest component is main
        everything else is extra
    """
    if not components:
        return None

    main = components[0]
    extras = components[1:]

    return {
        "view_type": "largest_component_view",
        "main_component_id": main["id"],
        "main_box": main["box"],
        "extra_component_ids": [comp["id"] for comp in extras],
        "component_count": len(components),
    }


def build_all_components_view(grid, components):
    """
    View 2:
        every component matters
        no main/blob distinction yet
    """
    return {
        "view_type": "all_components_view",
        "component_count": len(components),
        "component_ids": [comp["id"] for comp in components],
        "boxes": [comp["box"] for comp in components],
        "sizes": [comp["size"] for comp in components],
        "fill_ratios": [round(comp["fill_ratio"], 3) for comp in components],
    }


def assign_blobs_to_rings(rings, blobs):
    """
    For each blob:
        assign it to the smallest ring box that contains its center.

    If no ring contains it:
        outside_all
    """
    assignments = []

    for blob in blobs:
        blob_box = blob.get("box")
        center = box_center(blob_box)

        if center is None:
            assignments.append({
                "blob_id": blob["id"],
                "assigned_to": "unknown",
            })
            continue

        r, c = center

        containing_rings = []

        for ring in rings:
            ring_box = ring.get("box")

            if box_contains_point(ring_box, r, c):
                containing_rings.append(ring)

        if not containing_rings:
            assignments.append({
                "blob_id": blob["id"],
                "assigned_to": "outside_all",
            })
            continue

        # If multiple rings contain the blob, choose the smallest box.
        containing_rings.sort(
            key=lambda ring: ring["box"]["area"]
        )

        assignments.append({
            "blob_id": blob["id"],
            "assigned_to": containing_rings[0]["id"],
        })

    return assignments


def build_ring_blob_view(grid, components):
    """
    View 3:
        possible rings + possible blobs

    This is only one candidate interpretation.
    Later scoring decides whether this view is useful.
    """
    classified = classify_components_soft(components)

    rings = [
        comp
        for comp in classified
        if comp["soft_label"] == "possible_ring"
    ]

    blobs = [
        comp
        for comp in classified
        if comp["soft_label"] == "possible_blob"
    ]

    # Unknown components can be treated as blobs for this candidate view.
    for comp in classified:
        if comp["soft_label"] == "unknown":
            blobs.append(comp)

    assignments = assign_blobs_to_rings(rings, blobs)

    return {
        "view_type": "ring_blob_view",
        "ring_count": len(rings),
        "blob_count": len(blobs),
        "ring_ids": [ring["id"] for ring in rings],
        "blob_ids": [blob["id"] for blob in blobs],
        "ring_boxes": [ring["box"] for ring in rings],
        "blob_boxes": [blob["box"] for blob in blobs],
        "blob_assignments": assignments,
    }


def build_foreground_box_view(grid, components):
    """
    View 4:
        ignore components and just look at the whole foreground box.
    """
    fg_box = get_foreground_box(grid)

    return {
        "view_type": "foreground_box_view",
        "foreground_box": fg_box,
        "component_count": len(components),
    }


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def discover_visual_abstractions(grid):
    """
    Return multiple possible views of the same input grid.

    This is where learning begins:
        we do NOT pick a truth here.
        we only produce candidate ways of seeing.
    """
    if grid is None:
        return []

    h, w = grid_shape(grid)
    bg = get_background_color(grid)
    active = get_active_colors(grid)
    components = find_components(grid)

    views = []

    largest_view = build_largest_component_view(grid, components)
    if largest_view is not None:
        views.append(largest_view)

    views.append(build_all_components_view(grid, components))
    views.append(build_ring_blob_view(grid, components))
    views.append(build_foreground_box_view(grid, components))

    return {
        "grid_shape": (h, w),
        "background": bg,
        "active_colors": active,
        "components": components,
        "views": views,
    }


def print_visual_abstractions(summary):
    """
    Small debug printer for one input.
    """
    if summary is None:
        print("No visual abstraction summary.")
        return

    print("\nVISUAL ABSTRACTION SUMMARY")
    print("-" * 60)
    print(f"Grid shape   : {summary.get('grid_shape')}")
    print(f"Background   : {summary.get('background')}")
    print(f"Active colors: {summary.get('active_colors')}")
    print(f"Components   : {len(summary.get('components', []))}")

    print("\nComponents")
    print("-" * 60)

    for comp in summary.get("components", []):
        box = comp["box"]
        print(
            f"id={comp['id']} "
            f"size={comp['size']} "
            f"box=({box['height']}x{box['width']}) "
            f"fill={comp['fill_ratio']:.3f} "
            f"colors={comp['colors']}"
        )

    print("\nViews")
    print("-" * 60)

    for view in summary.get("views", []):
        print(f"{view['view_type']}: {view}")