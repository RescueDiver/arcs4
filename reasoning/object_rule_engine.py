from vision.object_finder import find_objects
from reasoning.object_selector import select_most_multicolor_object
from core.transforms import apply_transform
from core.scoring import score_prediction


def get_grid_border_colors(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    top = {v for v in grid[0] if v != 0}
    bottom = {v for v in grid[-1] if v != 0}
    left = {grid[r][0] for r in range(rows) if grid[r][0] != 0}
    right = {grid[r][cols - 1] for r in range(rows) if grid[r][cols - 1] != 0}

    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }


def get_visible_edge_colors(patch):
    h = len(patch)
    w = len(patch[0]) if h else 0

    top = set()
    bottom = set()
    left = set()
    right = set()

    for c in range(w):
        for r in range(h):
            if patch[r][c] != 0:
                top.add(patch[r][c])
                break

    for c in range(w):
        for r in range(h - 1, -1, -1):
            if patch[r][c] != 0:
                bottom.add(patch[r][c])
                break

    for r in range(h):
        for c in range(w):
            if patch[r][c] != 0:
                left.add(patch[r][c])
                break

    for r in range(h):
        for c in range(w - 1, -1, -1):
            if patch[r][c] != 0:
                right.add(patch[r][c])
                break

    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }


def score_edge_alignment(edge_colors, border_colors):
    score = 0
    for side in ["top", "bottom", "left", "right"]:
        score += len(edge_colors[side] & border_colors[side])
    return score


def peel_matching_outer_wrapper(patch, border_color_union):
    grid = [row[:] for row in patch]

    changed = True
    while changed:
        changed = False

        # top row
        if grid and all(v == 0 or v in border_color_union for v in grid[0]):
            grid = grid[1:]
            changed = True
            if not grid:
                return [[0]]

        # bottom row
        if grid and all(v == 0 or v in border_color_union for v in grid[-1]):
            grid = grid[:-1]
            changed = True
            if not grid:
                return [[0]]

        # left col
        if grid and all(row[0] == 0 or row[0] in border_color_union for row in grid):
            grid = [row[1:] for row in grid if len(row) > 1]
            changed = True
            if not grid or not grid[0]:
                return [[0]]

        # right col
        if grid and all(row[-1] == 0 or row[-1] in border_color_union for row in grid):
            grid = [row[:-1] for row in grid if len(row) > 1]
            changed = True
            if not grid or not grid[0]:
                return [[0]]

    return grid


def solve_pair_fc7_rule(input_grid, output_grid):
    objects = find_objects(input_grid)
    if not objects:
        return None

    obj = select_most_multicolor_object(objects)
    if obj is None:
        return None

    border_colors = get_grid_border_colors(input_grid)
    border_union = set().union(*border_colors.values())

    best = None
    best_score = -10**9

    for transform in [
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "flip_horizontal",
    "flip_vertical",
    "rotate_90_flip_horizontal",
    "rotate_270_flip_horizontal",
]:
        rotated = apply_transform(obj["patch"], transform)

        edge_colors = get_visible_edge_colors(rotated)
        align_score = score_edge_alignment(edge_colors, border_colors)

        peeled = peel_matching_outer_wrapper(rotated, border_union)
        score = score_prediction(peeled, output_grid)

        total_score = score + align_score * 10

        candidate = {
            "selector": "most_multicolor_object",
            "transform": transform,
            "object": obj,
            "predicted": peeled,
            "score": total_score,
            "exact": peeled == output_grid,
            "align_score": align_score,
        }

        if total_score > best_score:
            best_score = total_score
            best = candidate

    return best