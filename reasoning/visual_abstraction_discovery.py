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
# Important:
#   We do not force one interpretation.
#
#   We generate candidate views:
#       - largest component view
#       - all components view
#       - ring/blob view
#       - foreground box view
#       - enclosure tree view
#
# New idea:
#   Humans often see visual scenes outside-to-inside:
#
#       whole grid size
#       -> background / foreground
#       -> largest meaningful structure
#       -> enclosures / containers
#       -> nested enclosures
#       -> blobs/details
#
#   The enclosure_tree_view is meant to capture that.
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
    """
    Default background guess:
        most common color.

    This is still useful for old views.
    """
    counts = color_counts(grid)

    if not counts:
        return 0

    return counts.most_common(1)[0][0]


def get_edge_background_color(grid):
    """
    Better visual background guess:
        most common color touching the outer edge.

    This matches the outside-to-inside idea better.
    """
    if grid is None:
        return 0

    h, w = grid_shape(grid)

    if h == 0 or w == 0:
        return 0

    counts = Counter()

    for c in range(w):
        counts[grid[0][c]] += 1
        counts[grid[h - 1][c]] += 1

    for r in range(1, h - 1):
        counts[grid[r][0]] += 1
        counts[grid[r][w - 1]] += 1

    if not counts:
        return get_background_color(grid)

    return counts.most_common(1)[0][0]


def get_active_colors(grid):
    bg = get_background_color(grid)
    counts = color_counts(grid)

    return [
        color
        for color, count in counts.most_common()
        if color != bg
    ]


def get_foreground_colors_by_edge_bg(grid):
    bg = get_edge_background_color(grid)
    counts = color_counts(grid)

    return [
        color
        for color, count in counts.most_common()
        if color != bg
    ]


def in_bounds(grid, r, c):
    h, w = grid_shape(grid)
    return 0 <= r < h and 0 <= c < w


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


def box_strictly_contains_box(outer_box, inner_box):
    """
    Strict containment means the inner box is inside the outer box,
    but not exactly the same box.
    """
    if not box_contains_box(outer_box, inner_box):
        return False

    return outer_box != inner_box


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


def box_area(box):
    if box is None:
        return 0

    return box.get("area", 0)


def box_perimeter_estimate(box):
    if box is None:
        return 0

    h = box["height"]
    w = box["width"]

    if h <= 0 or w <= 0:
        return 0

    if h == 1:
        return w

    if w == 1:
        return h

    return 2 * h + 2 * w - 4


def box_sort_key_largest_first(item):
    box = item.get("box")

    return (
        -box_area(box),
        item.get("id", 999999),
    )


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

def find_components(grid):
    """
    Find 4-connected non-background components.

    Uses most-common color as background.

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

            box_area_value = max(1, box["area"])
            fill_ratio = len(cells) / box_area_value

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


def find_components_by_color_rule(grid, include_cell_func):
    """
    Generic 4-connected component finder.

    include_cell_func(value, r, c) -> True/False
    """
    if grid is None:
        return []

    h, w = grid_shape(grid)
    seen = set()
    components = []

    for sr in range(h):
        for sc in range(w):
            if (sr, sc) in seen:
                continue

            if not include_cell_func(grid[sr][sc], sr, sc):
                continue

            q = deque([(sr, sc)])
            seen.add((sr, sc))

            cells = []
            colors = set()

            while q:
                r, c = q.popleft()
                value = grid[r][c]

                cells.append((r, c, value))
                colors.add(value)

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = r + dr
                    nc = c + dc

                    if not in_bounds(grid, nr, nc):
                        continue

                    if (nr, nc) in seen:
                        continue

                    if not include_cell_func(grid[nr][nc], nr, nc):
                        continue

                    seen.add((nr, nc))
                    q.append((nr, nc))

            box = make_box_from_cells(cells)

            if box is None:
                continue

            fill_ratio = len(cells) / max(1, box["area"])

            components.append({
                "id": len(components),
                "cells": cells,
                "colors": tuple(sorted(colors)),
                "size": len(cells),
                "box": box,
                "fill_ratio": fill_ratio,
            })

    components.sort(key=lambda comp: comp["size"], reverse=True)

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
# OLD ABSTRACTION VIEW BUILDERS
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
        "rings": rings,
        "blobs": blobs,
        "ring_boxes": [ring["box"] for ring in rings],
        "blob_boxes": [blob["box"] for blob in blobs],
        "assignments": assignments,
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
# ENCLOSURE DISCOVERY HELPERS
# ============================================================

def flood_fill_background_regions(grid, background_color):
    """
    Split background into connected regions.

    The outside background is the region touching the grid edge.

    Any background region that does NOT touch the grid edge is enclosed
    or almost enclosed by foreground.
    """
    h, w = grid_shape(grid)

    seen = set()
    regions = []

    for sr in range(h):
        for sc in range(w):
            if (sr, sc) in seen:
                continue

            if grid[sr][sc] != background_color:
                continue

            q = deque([(sr, sc)])
            seen.add((sr, sc))

            cells = []
            touches_edge = False

            while q:
                r, c = q.popleft()
                cells.append((r, c))

                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_edge = True

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = r + dr
                    nc = c + dc

                    if not in_bounds(grid, nr, nc):
                        continue

                    if (nr, nc) in seen:
                        continue

                    if grid[nr][nc] != background_color:
                        continue

                    seen.add((nr, nc))
                    q.append((nr, nc))

            box = make_box_from_cells(cells)

            regions.append({
                "id": len(regions),
                "cells": cells,
                "cell_count": len(cells),
                "box": box,
                "touches_edge": touches_edge,
            })

    regions.sort(
        key=lambda region: (
            region["touches_edge"],
            -region["cell_count"],
        )
    )

    for idx, region in enumerate(regions):
        region["id"] = idx

    return regions


def get_neighbor_foreground_cells(grid, region_cells, background_color):
    """
    For one background region, find foreground cells touching it.

    These are likely the boundary/wall cells around that region.
    """
    neighbor_cells = set()

    for r, c in region_cells:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = r + dr
            nc = c + dc

            if not in_bounds(grid, nr, nc):
                continue

            if grid[nr][nc] != background_color:
                neighbor_cells.add((nr, nc, grid[nr][nc]))

    return list(neighbor_cells)


def estimate_boundary_completeness(grid, inside_region, background_color):
    """
    Estimate how complete the wall around an enclosed background region is.

    This is not perfect geometry.
    It is a scale-aware signal.

    For true enclosed background regions:
        leak_exists = False
        enclosure_state = closed_enclosure

    Later we can improve almost-closed detection by intentionally testing
    small wall gaps.
    """
    inside_cells = inside_region.get("cells", [])
    wall_cells = get_neighbor_foreground_cells(
        grid,
        inside_cells,
        background_color,
    )

    wall_box = make_box_from_cells(wall_cells)

    if wall_box is None:
        return {
            "wall_box": None,
            "wall_cell_count": 0,
            "estimated_perimeter": 0,
            "wall_completeness": 0.0,
            "leak_exists": True,
            "gap_count": None,
            "gap_ratio_perimeter": None,
            "state": "open_shape_not_enclosure",
        }

    estimated_perimeter = max(1, box_perimeter_estimate(wall_box))

    # Boundary cells can be more than perimeter for thick/irregular walls.
    completeness = min(1.0, len(wall_cells) / estimated_perimeter)

    return {
        "wall_box": wall_box,
        "wall_cell_count": len(wall_cells),
        "estimated_perimeter": estimated_perimeter,
        "wall_completeness": round(completeness, 3),
        "leak_exists": False,
        "gap_count": 0,
        "gap_ratio_perimeter": 0.0,
        "state": "closed_enclosure",
    }


def find_foreground_components_inside_box(grid, box, background_color):
    """
    Find foreground components whose centers are inside a box.
    """
    if box is None:
        return []

    components = find_components_by_color_rule(
        grid,
        lambda value, r, c: (
            value != background_color
            and box_contains_point(box, r, c)
        ),
    )

    return components


def is_component_boundary_for_region(component, region_boundary_cells):
    """
    True if this foreground component overlaps the boundary/wall cells.
    """
    boundary_positions = {
        (r, c)
        for r, c, value in region_boundary_cells
    }

    for r, c, value in component.get("cells", []):
        if (r, c) in boundary_positions:
            return True

    return False


def component_to_blob_node(component):
    box = component.get("box")

    return {
        "node_type": "blob",
        "id": f"blob_{component.get('id')}",
        "component_id": component.get("id"),
        "box": box,
        "size": component.get("size", 0),
        "colors": component.get("colors", ()),
        "fill_ratio": round(component.get("fill_ratio", 0.0), 3),
        "children": [],
    }


def enclosure_to_node(enclosure):
    return {
        "node_type": "enclosure",
        "id": enclosure["id"],
        "enclosure_index": enclosure["enclosure_index"],
        "state": enclosure["state"],
        "box": enclosure["box"],
        "inside_box": enclosure["inside_box"],
        "inside_area": enclosure["inside_area"],
        "wall_box": enclosure["wall_box"],
        "wall_cell_count": enclosure["wall_cell_count"],
        "wall_completeness": enclosure["wall_completeness"],
        "leak_exists": enclosure["leak_exists"],
        "gap_count": enclosure["gap_count"],
        "gap_ratio_perimeter": enclosure["gap_ratio_perimeter"],
        "children": [],
    }


def find_enclosed_background_enclosures(grid, background_color):
    """
    Find closed enclosures by finding background regions trapped away
    from the outside background.

    This is the first version of true outside-to-inside detection.
    """
    background_regions = flood_fill_background_regions(
        grid,
        background_color,
    )

    enclosed_regions = [
        region
        for region in background_regions
        if not region.get("touches_edge", False)
    ]

    enclosures = []

    for idx, region in enumerate(enclosed_regions):
        boundary_cells = get_neighbor_foreground_cells(
            grid,
            region.get("cells", []),
            background_color,
        )

        boundary_info = estimate_boundary_completeness(
            grid,
            region,
            background_color,
        )

        wall_box = boundary_info["wall_box"]
        inside_box = region.get("box")

        # The visual enclosure box should include both the trapped
        # background and the wall around it.
        combined_cells = []

        for r, c in region.get("cells", []):
            combined_cells.append((r, c))

        for r, c, value in boundary_cells:
            combined_cells.append((r, c))

        enclosure_box = make_box_from_cells(combined_cells)

        enclosures.append({
            "id": f"enclosure_{idx}",
            "enclosure_index": idx,
            "state": boundary_info["state"],
            "box": enclosure_box,
            "inside_box": inside_box,
            "inside_area": region.get("cell_count", 0),
            "inside_region_id": region.get("id"),
            "inside_cells": region.get("cells", []),
            "boundary_cells": boundary_cells,
            "wall_box": wall_box,
            "wall_cell_count": boundary_info["wall_cell_count"],
            "wall_completeness": boundary_info["wall_completeness"],
            "leak_exists": boundary_info["leak_exists"],
            "gap_count": boundary_info["gap_count"],
            "gap_ratio_perimeter": boundary_info["gap_ratio_perimeter"],
        })

    enclosures.sort(key=box_sort_key_largest_first)

    for idx, enclosure in enumerate(enclosures):
        enclosure["id"] = f"enclosure_{idx}"
        enclosure["enclosure_index"] = idx

    return enclosures


def choose_parent_enclosure(enclosure, all_enclosures):
    """
    Parent is the smallest larger enclosure that contains this enclosure.
    """
    box = enclosure.get("box")

    candidates = []

    for other in all_enclosures:
        if other is enclosure:
            continue

        other_box = other.get("box")

        if box_strictly_contains_box(other_box, box):
            candidates.append(other)

    if not candidates:
        return None

    candidates.sort(key=lambda item: box_area(item.get("box")))

    return candidates[0]


def build_enclosure_parent_map(enclosures):
    parent = {}

    for enclosure in enclosures:
        parent_id = None
        parent_enclosure = choose_parent_enclosure(enclosure, enclosures)

        if parent_enclosure is not None:
            parent_id = parent_enclosure["id"]

        parent[enclosure["id"]] = parent_id

    return parent


def find_direct_blob_children_for_enclosure(grid, enclosure, enclosures, background_color):
    """
    Find blob children directly inside this enclosure.

    Important:
        Use GLOBAL foreground components, not cropped-inside-box components.

    Why:
        If we search only inside the enclosure box, wall pieces get chopped up
        and falsely counted as blobs.
    """
    inside_box = enclosure.get("inside_box")

    if inside_box is None:
        return []

    # Use full-grid components so enclosure walls stay whole.
    global_components = find_components_by_color_rule(
        grid,
        lambda value, r, c: value != background_color,
    )

    # Boundary cells for this enclosure.
    this_boundary_cells = {
        (r, c)
        for r, c, value in enclosure.get("boundary_cells", [])
    }

    # Boundary cells for nested child enclosures.
    child_enclosure_boxes = []
    child_boundary_cells = set()

    for other in enclosures:
        if other["id"] == enclosure["id"]:
            continue

        other_box = other.get("box")

        if box_strictly_contains_box(enclosure.get("box"), other_box):
            child_enclosure_boxes.append(other_box)

            for r, c, value in other.get("boundary_cells", []):
                child_boundary_cells.add((r, c))

    blobs = []

    for comp in global_components:
        comp_box = comp.get("box")

        if comp_box is None:
            continue

        center = box_center(comp_box)

        if center is None:
            continue

        cr, cc = center

        # Blob center must be inside this enclosure's open interior.
        if not box_contains_point(inside_box, cr, cc):
            continue

        comp_cells = {
            (r, c)
            for r, c, value in comp.get("cells", [])
        }

        # Do not count this enclosure wall as a blob.
        if comp_cells & this_boundary_cells:
            continue

        # Do not count nested enclosure walls as blobs.
        if comp_cells & child_boundary_cells:
            continue

        # Do not attach blobs directly to this enclosure
        # if their center is inside a smaller nested enclosure.
        inside_nested_child = False

        for child_box in child_enclosure_boxes:
            if box_contains_point(child_box, cr, cc):
                inside_nested_child = True
                break

        if inside_nested_child:
            continue

        blobs.append(comp)

    return blobs


def attach_children_to_enclosure_nodes(grid, enclosures, background_color):
    """
    Build the actual nested scene tree.
    """
    parent_map = build_enclosure_parent_map(enclosures)

    nodes = {
        enclosure["id"]: enclosure_to_node(enclosure)
        for enclosure in enclosures
    }

    root = {
        "node_type": "root",
        "id": "ROOT",
        "children": [],
    }

    # Attach enclosure nodes to root or to their parent enclosure.
    for enclosure in enclosures:
        node = nodes[enclosure["id"]]
        parent_id = parent_map[enclosure["id"]]

        if parent_id is None:
            root["children"].append(node)
        else:
            nodes[parent_id]["children"].append(node)

    # Attach direct blob children to each enclosure.
    for enclosure in enclosures:
        node = nodes[enclosure["id"]]

        direct_blobs = find_direct_blob_children_for_enclosure(
            grid,
            enclosure,
            enclosures,
            background_color,
        )

        for blob_comp in direct_blobs:
            node["children"].append(component_to_blob_node(blob_comp))

    # Keep children ordered large-to-small, enclosure before blob.
    def child_sort_key(child):
        node_type = child.get("node_type")

        type_order = {
            "enclosure": 0,
            "blob": 1,
        }.get(node_type, 9)

        return (
            type_order,
            -box_area(child.get("box")),
            child.get("id", ""),
        )

    def sort_tree(node):
        node["children"].sort(key=child_sort_key)

        for child in node["children"]:
            sort_tree(child)

    sort_tree(root)

    return root, parent_map


def count_tree_nodes_by_type(node, node_type):
    if node is None:
        return 0

    count = 1 if node.get("node_type") == node_type else 0

    for child in node.get("children", []):
        count += count_tree_nodes_by_type(child, node_type)

    return count


def tree_max_depth(node):
    if node is None:
        return 0

    children = node.get("children", [])

    if not children:
        return 1

    return 1 + max(tree_max_depth(child) for child in children)


def flatten_tree(node):
    if node is None:
        return []

    out = [node]

    for child in node.get("children", []):
        out.extend(flatten_tree(child))

    return out


def build_enclosure_tree_view(grid):
    """
    New view:
        outside-to-inside human-style scene tree.

    First version detects closed enclosures by trapped background regions.

    Later version can add:
        - almost_closed_enclosure
        - gap metadata
        - leak width
        - gap alignment
    """
    if grid is None:
        return {
            "view_type": "enclosure_tree_view",
            "grid_shape": (0, 0),
            "background": 0,
            "foreground_colors": [],
            "enclosure_count": 0,
            "blob_count": 0,
            "tree_depth": 0,
            "tree": {
                "node_type": "root",
                "id": "ROOT",
                "children": [],
            },
            "enclosures": [],
            "parent_map": {},
        }

    h, w = grid_shape(grid)

    background = get_edge_background_color(grid)
    foreground_colors = get_foreground_colors_by_edge_bg(grid)

    enclosures = find_enclosed_background_enclosures(
        grid,
        background,
    )

    tree, parent_map = attach_children_to_enclosure_nodes(
        grid,
        enclosures,
        background,
    )

    return {
        "view_type": "enclosure_tree_view",

        "grid_shape": (h, w),
        "background": background,
        "foreground_colors": foreground_colors,

        "enclosure_count": len(enclosures),
        "blob_count": count_tree_nodes_by_type(tree, "blob"),
        "tree_depth": tree_max_depth(tree),

        "tree": tree,
        "enclosures": enclosures,
        "parent_map": parent_map,
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
    edge_bg = get_edge_background_color(grid)
    active = get_active_colors(grid)
    foreground_colors = get_foreground_colors_by_edge_bg(grid)
    components = find_components(grid)

    views = []

    largest_view = build_largest_component_view(grid, components)
    if largest_view is not None:
        views.append(largest_view)

    views.append(build_all_components_view(grid, components))
    views.append(build_ring_blob_view(grid, components))
    views.append(build_foreground_box_view(grid, components))

    # New human-style view.
    views.append(build_enclosure_tree_view(grid))

    return {
        "grid_shape": (h, w),
        "background": bg,
        "edge_background": edge_bg,
        "active_colors": active,
        "foreground_colors": foreground_colors,
        "components": components,
        "views": views,
    }


# ============================================================
# DEBUG PRINTING
# ============================================================

def print_tree_node(node, indent=0):
    prefix = "    " * indent

    node_type = node.get("node_type")
    node_id = node.get("id")

    if node_type == "root":
        print(f"{prefix}{node_id}")

    elif node_type == "enclosure":
        box = node.get("box")
        print(
            f"{prefix}└── {node_id} "
            f"state={node.get('state')} "
            f"box={box} "
            f"wall={node.get('wall_completeness')}"
        )

    elif node_type == "blob":
        box = node.get("box")
        print(
            f"{prefix}└── {node_id} "
            f"size={node.get('size')} "
            f"box={box} "
            f"colors={node.get('colors')}"
        )

    else:
        print(f"{prefix}└── {node_type}:{node_id}")

    for child in node.get("children", []):
        print_tree_node(child, indent + 1)


def print_visual_abstractions(summary):
    """
    Small debug printer for one input.

    This file can print for standalone debugging,
    but solver/router code should not depend on printing.
    """
    if summary is None:
        print("No visual abstraction summary.")
        return

    print("\nVISUAL ABSTRACTION SUMMARY")
    print("-" * 60)
    print(f"Grid shape     : {summary.get('grid_shape')}")
    print(f"Background     : {summary.get('background')}")
    print(f"Edge background: {summary.get('edge_background')}")
    print(f"Active colors  : {summary.get('active_colors')}")
    print(f"Foreground     : {summary.get('foreground_colors')}")
    print(f"Components     : {len(summary.get('components', []))}")

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
        view_type = view.get("view_type")

        if view_type != "enclosure_tree_view":
            print(f"{view_type}: {view}")
            continue

        print("enclosure_tree_view:")
        print(f"  enclosure_count: {view.get('enclosure_count')}")
        print(f"  blob_count     : {view.get('blob_count')}")
        print(f"  tree_depth     : {view.get('tree_depth')}")
        print_tree_node(view.get("tree", {}))