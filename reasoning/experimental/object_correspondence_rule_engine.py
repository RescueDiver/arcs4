from vision.object_finder import find_objects
from core.transforms import apply_transform
from core.scoring import score_prediction
from debug.debug_utils import debug_selected_object


TRANSFORMS = [
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "flip_horizontal",
    "flip_vertical",
    "rotate_90_flip_horizontal",
    "rotate_270_flip_horizontal",
]


def touches_border(obj, rows, cols):
    bbox = obj["bbox"]
    return (
        bbox["min_r"] == 0
        or bbox["min_c"] == 0
        or bbox["max_r"] == rows - 1
        or bbox["max_c"] == cols - 1
    )


def is_full_grid_object(obj, rows, cols):
    bbox = obj["bbox"]
    return (
        bbox["min_r"] == 0
        and bbox["min_c"] == 0
        and bbox["max_r"] == rows - 1
        and bbox["max_c"] == cols - 1
    )


def select_object_by_rule(objects, selector_name, rows, cols):
    if not objects:
        return None

    if selector_name == "largest_object":
        return max(objects, key=lambda o: o["area"])

    if selector_name == "smallest_object":
        return min(objects, key=lambda o: o["area"])

    if selector_name == "most_colors":
        return max(objects, key=lambda o: (o["color_count"], o["area"]))

    if selector_name == "least_colors":
        return min(objects, key=lambda o: (o["color_count"], o["area"]))

    if selector_name == "touches_border":
        border_objs = [o for o in objects if touches_border(o, rows, cols)]
        if not border_objs:
            return None
        return max(border_objs, key=lambda o: o["area"])

    if selector_name == "not_touches_border":
        inner_objs = [o for o in objects if not touches_border(o, rows, cols)]
        if not inner_objs:
            return None
        return max(inner_objs, key=lambda o: o["area"])

    if selector_name == "largest_non_full_grid":
        candidates = [o for o in objects if not is_full_grid_object(o, rows, cols)]
        if not candidates:
            return None
        return max(candidates, key=lambda o: o["area"])

    if selector_name == "most_colors_non_full_grid":
        candidates = [o for o in objects if not is_full_grid_object(o, rows, cols)]
        if not candidates:
            return None
        return max(candidates, key=lambda o: (o["color_count"], o["area"]))

    return None


def normalize_shape(grid):
    return [[1 if cell != 0 else 0 for cell in row] for row in grid]


def same_shape_ignore_color(a, b):
    if len(a) != len(b):
        return False
    if not a and not b:
        return True
    if not a or not b:
        return False
    if len(a[0]) != len(b[0]):
        return False

    na = normalize_shape(a)
    nb = normalize_shape(b)

    for r in range(len(na)):
        for c in range(len(na[0])):
            if na[r][c] != nb[r][c]:
                return False

    return True


def infer_recolor_map(src, dst):
    if len(src) != len(dst):
        return None
    if not src and not dst:
        return {}
    if not src or not dst:
        return None
    if len(src[0]) != len(dst[0]):
        return None

    mapping = {}

    for r in range(len(src)):
        for c in range(len(src[0])):
            s = src[r][c]
            d = dst[r][c]

            if s == 0 and d != 0:
                return None

            if s != 0:
                if s in mapping and mapping[s] != d:
                    return None
                mapping[s] = d

    return mapping


def apply_recolor_map(grid, recolor_map):
    if recolor_map is None:
        return [row[:] for row in grid]

    out = []
    for row in grid:
        new_row = []
        for cell in row:
            if cell == 0:
                new_row.append(0)
            else:
                new_row.append(recolor_map.get(cell, cell))
        out.append(new_row)
    return out


def solve_pair_object_correspondence_rule(input_grid, output_grid):
    objects = find_objects(input_grid)
    if not objects:
        return None

    rows = len(input_grid)
    cols = len(input_grid[0]) if rows else 0

    selectors = [
        "largest_object",
        "smallest_object",
        "most_colors",
        "least_colors",
        "touches_border",
        "not_touches_border",
        "largest_non_full_grid",
        "most_colors_non_full_grid",
    ]

    best = None
    best_score = -10**9

    for selector in selectors:
        obj = select_object_by_rule(objects, selector, rows, cols)
        if obj is None:
            continue

        debug_selected_object(f"CORRESPONDENCE OBJECT ({selector})", obj)

        base_patch = obj["patch"]

        for transform in TRANSFORMS:
            try:
                transformed = apply_transform(base_patch, transform)
            except Exception:
                continue

            recolor_map = None
            predicted = transformed
            shape_match = same_shape_ignore_color(transformed, output_grid)

            # exact first
            score = score_prediction(predicted, output_grid)
            exact = predicted == output_grid

            # if not exact, try recolor only when shapes match
            if not exact and shape_match:
                maybe_map = infer_recolor_map(transformed, output_grid)
                if maybe_map is not None:
                    recolored = apply_recolor_map(transformed, maybe_map)
                    recolored_score = score_prediction(recolored, output_grid)

                    if recolored_score > score:
                        predicted = recolored
                        recolor_map = maybe_map
                        score = recolored_score
                        exact = predicted == output_grid

            candidate = {
                "selector": selector,
                "transform": transform,
                "recolor_map": recolor_map,
                "selected_object": obj,
                "object": obj,
                "predicted": predicted,
                "score": score,
                "exact": exact,
            }

            if score > best_score:
                best_score = score
                best = candidate

    return best