# reasoning/ring_detector.py
"""
Ring Detector

Goal:
    Teach the solver to see "rings" the way a human sees them.

A ring is:
    foreground cells that create an enclosed interior background region.

This file does not solve the ARC task.
It only discovers ring-like structures from a grid.
"""

from collections import deque


def grid_shape(grid):
    if not grid:
        return (0, 0)

    return (len(grid), len(grid[0]))


def in_bounds(grid, r, c):
    h, w = grid_shape(grid)
    return 0 <= r < h and 0 <= c < w


def neighbors4(r, c):
    return [
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
    ]


def get_color_counts(grid):
    counts = {}

    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1

    return counts


def get_background_color(grid):
    """
    Usually ARC background is the most common color.
    """
    counts = get_color_counts(grid)

    if not counts:
        return 0

    return max(counts, key=counts.get)


def get_foreground_colors(grid, background_color=None):
    """
    Foreground colors are every color except the background.
    """
    if background_color is None:
        background_color = get_background_color(grid)

    colors = sorted(get_color_counts(grid).keys())

    return [
        color for color in colors
        if color != background_color
    ]


def connected_components_for_color(grid, color):
    """
    Find connected components made of one color.
    """
    h, w = grid_shape(grid)
    visited = set()
    components = []

    for r in range(h):
        for c in range(w):
            if (r, c) in visited:
                continue

            if grid[r][c] != color:
                continue

            queue = deque([(r, c)])
            visited.add((r, c))
            cells = []

            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))

                for nr, nc in neighbors4(cr, cc):
                    if not in_bounds(grid, nr, nc):
                        continue

                    if (nr, nc) in visited:
                        continue

                    if grid[nr][nc] != color:
                        continue

                    visited.add((nr, nc))
                    queue.append((nr, nc))

            components.append(cells)

    return components


def bbox_for_cells(cells):
    rows = [r for r, c in cells]
    cols = [c for r, c in cells]

    return (
        min(rows),
        min(cols),
        max(rows),
        max(cols),
    )


def center_for_cells(cells):
    if not cells:
        return None

    row_sum = sum(r for r, c in cells)
    col_sum = sum(c for r, c in cells)

    return (
        row_sum / len(cells),
        col_sum / len(cells),
    )


def flood_fill_outside_background(grid, background_color):
    """
    Mark all background cells connected to the outside border.

    Any background cell not reached by this is enclosed.
    """
    h, w = grid_shape(grid)
    outside = set()
    queue = deque()

    # Add all background cells on the outer border.
    for r in range(h):
        for c in [0, w - 1]:
            if grid[r][c] == background_color and (r, c) not in outside:
                outside.add((r, c))
                queue.append((r, c))

    for c in range(w):
        for r in [0, h - 1]:
            if grid[r][c] == background_color and (r, c) not in outside:
                outside.add((r, c))
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()

        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid, nr, nc):
                continue

            if (nr, nc) in outside:
                continue

            if grid[nr][nc] != background_color:
                continue

            outside.add((nr, nc))
            queue.append((nr, nc))

    return outside


def find_enclosed_background_regions(grid, background_color=None):
    """
    Find background regions that are not connected to the outside.

    These are the interior spaces inside rings.
    """
    if background_color is None:
        background_color = get_background_color(grid)

    h, w = grid_shape(grid)
    outside_background = flood_fill_outside_background(
        grid,
        background_color,
    )

    visited = set(outside_background)
    enclosed_regions = []

    for r in range(h):
        for c in range(w):
            if (r, c) in visited:
                continue

            if grid[r][c] != background_color:
                continue

            queue = deque([(r, c)])
            visited.add((r, c))
            cells = []

            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))

                for nr, nc in neighbors4(cr, cc):
                    if not in_bounds(grid, nr, nc):
                        continue

                    if (nr, nc) in visited:
                        continue

                    if grid[nr][nc] != background_color:
                        continue

                    visited.add((nr, nc))
                    queue.append((nr, nc))

            enclosed_regions.append(cells)

    return enclosed_regions


def foreground_cells_touching_region(grid, region_cells, background_color):
    """
    Find foreground cells directly touching an enclosed background region.
    These are the ring wall cells around that interior.
    """
    touching = set()

    for r, c in region_cells:
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid, nr, nc):
                continue

            if grid[nr][nc] != background_color:
                touching.add((nr, nc))

    return touching


def find_ring_components(grid, background_color=None):
    """
    Discover rings by looking for enclosed background regions.

    Output:
        list of ring dictionaries:
            {
                "ring_index": 0,
                "color": 9,
                "interior_cells": [...],
                "interior_bbox": (...),
                "wall_touching_cells": [...],
                "wall_bbox": (...),
                "center": (...),
            }
    """
    if background_color is None:
        background_color = get_background_color(grid)

    enclosed_regions = find_enclosed_background_regions(
        grid,
        background_color,
    )

    rings = []

    for region_index, interior_cells in enumerate(enclosed_regions):
        touching_wall_cells = foreground_cells_touching_region(
            grid,
            interior_cells,
            background_color,
        )

        if not touching_wall_cells:
            continue

        wall_colors = {}

        for r, c in touching_wall_cells:
            color = grid[r][c]
            wall_colors[color] = wall_colors.get(color, 0) + 1

        ring_color = max(wall_colors, key=wall_colors.get)

        ring = {
            "ring_index": len(rings),
            "color": ring_color,
            "interior_cells": sorted(interior_cells),
            "interior_cell_count": len(interior_cells),
            "interior_bbox": bbox_for_cells(interior_cells),
            "wall_touching_cells": sorted(touching_wall_cells),
            "wall_touching_cell_count": len(touching_wall_cells),
            "wall_bbox": bbox_for_cells(touching_wall_cells),
            "center": center_for_cells(interior_cells),
        }

        rings.append(ring)

    return rings


def print_ring_summary(grid):
    """
    Debug helper.
    """
    background_color = get_background_color(grid)
    rings = find_ring_components(grid, background_color)

    print()
    print("RING DETECTOR SUMMARY")
    print("-" * 60)
    print(f"background color: {background_color}")
    print(f"ring count      : {len(rings)}")

    for ring in rings:
        print()
        print(f"Ring {ring['ring_index']}")
        print(f"  color               : {ring['color']}")
        print(f"  interior cells      : {ring['interior_cell_count']}")
        print(f"  interior bbox       : {ring['interior_bbox']}")
        print(f"  wall touching cells : {ring['wall_touching_cell_count']}")
        print(f"  wall bbox           : {ring['wall_bbox']}")
        print(f"  center              : {ring['center']}")

    return rings