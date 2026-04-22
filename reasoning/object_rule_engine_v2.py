from vision.object_finder import find_objects
from reasoning.object_selector import select_most_multicolor_nonframe_object
from core.transforms import apply_transform
from core.scoring import score_prediction
from debug.debug_utils import debug_selected_object
from reasoning.object_rule_engine import (
    get_grid_border_colors,
    get_visible_edge_colors,
    score_edge_alignment,
    peel_matching_outer_wrapper,
)


def solve_pair_object_rule_v2(input_grid, output_grid):
    objects = find_objects(input_grid)
    if not objects:
        return None

    rows = len(input_grid)
    cols = len(input_grid[0]) if rows else 0

    obj = select_most_multicolor_nonframe_object(objects, rows, cols)
    if obj is None:
        return None

    debug_selected_object("V2 OBJECT", obj)

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
            "selector": "most_multicolor_nonframe_object",
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