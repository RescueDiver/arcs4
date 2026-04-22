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

H_ALIGNS = ["left", "center", "right"]
V_ALIGNS = ["top", "middle", "bottom"]

SELECTORS = [
    "largest_non_full_grid",
    "most_colors_non_full_grid",
    "largest_object",
    "most_colors",
    "smallest_object",
    "least_colors",
    "touches_border",
    "not_touches_border",
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
        candidates = [o for o in objects if touches_border(o, rows, cols)]
        if not candidates:
            return None
        return max(candidates, key=lambda o: o["area"])

    if selector_name == "not_touches_border":
        candidates = [o for o in objects if not touches_border(o, rows, cols)]
        if not candidates:
            return None
        return max(candidates, key=lambda o: o["area"])

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


def make_blank_grid(h, w, fill=0):
    return [[fill for _ in range(w)] for _ in range(h)]


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


def get_anchor_start(container_size, content_size, align_name):
    if content_size > container_size:
        return None

    if align_name in ("top", "left"):
        return 0

    if align_name in ("middle", "center"):
        return (container_size - content_size) // 2

    if align_name in ("bottom", "right"):
        return container_size - content_size

    return 0


def project_patch_to_canvas(patch, out_h, out_w, v_align, h_align):
    ph = len(patch)
    pw = len(patch[0]) if ph else 0

    if ph == 0 or pw == 0:
        return None

    start_r = get_anchor_start(out_h, ph, v_align)
    start_c = get_anchor_start(out_w, pw, h_align)

    if start_r is None or start_c is None:
        return None

    canvas = make_blank_grid(out_h, out_w, fill=0)

    for r in range(ph):
        for c in range(pw):
            val = patch[r][c]
            if val != 0:
                canvas[start_r + r][start_c + c] = val

    return canvas


def same_shape_mask(a, b):
    if len(a) != len(b):
        return False
    if not a and not b:
        return True
    if not a or not b:
        return False
    if len(a[0]) != len(b[0]):
        return False

    for r in range(len(a)):
        for c in range(len(a[0])):
            if (a[r][c] != 0) != (b[r][c] != 0):
                return False
    return True


def try_projected_candidate(base_patch, output_grid, transform, v_align, h_align):
    transformed = apply_transform(base_patch, transform)

    out_h = len(output_grid)
    out_w = len(output_grid[0]) if out_h else 0

    projected = project_patch_to_canvas(
        transformed,
        out_h=out_h,
        out_w=out_w,
        v_align=v_align,
        h_align=h_align,
    )
    if projected is None:
        return None

    recolor_map = None
    predicted = projected

    exact = predicted == output_grid
    score = score_prediction(predicted, output_grid)

    if not exact and same_shape_mask(projected, output_grid):
        maybe_map = infer_recolor_map(projected, output_grid)
        if maybe_map is not None:
            recolored = apply_recolor_map(projected, maybe_map)
            recolored_score = score_prediction(recolored, output_grid)
            if recolored_score >= score:
                predicted = recolored
                recolor_map = maybe_map
                score = recolored_score
                exact = predicted == output_grid

    return {
        "predicted": predicted,
        "transform": transform,
        "v_align": v_align,
        "h_align": h_align,
        "recolor_map": recolor_map,
        "score": score,
        "exact": exact,
    }


def solve_pair_object_projection_rule(input_grid, output_grid):
    objects = find_objects(input_grid)
    if not objects:
        return None

    rows = len(input_grid)
    cols = len(input_grid[0]) if rows else 0

    best = None
    best_score = -10**9

    for selector in SELECTORS:
        obj = select_object_by_rule(objects, selector, rows, cols)
        if obj is None:
            continue

        debug_selected_object(f"PROJECTION OBJECT ({selector})", obj)

        base_patch = obj["patch"]
        if not base_patch:
            continue

        for transform in TRANSFORMS:
            for v_align in V_ALIGNS:
                for h_align in H_ALIGNS:
                    candidate_core = try_projected_candidate(
                        base_patch=base_patch,
                        output_grid=output_grid,
                        transform=transform,
                        v_align=v_align,
                        h_align=h_align,
                    )
                    if candidate_core is None:
                        continue

                    candidate = {
                        "strategy": "object_projection_rule",
                        "selector": selector,
                        "transform": candidate_core["transform"],
                        "v_align": candidate_core["v_align"],
                        "h_align": candidate_core["h_align"],
                        "recolor_map": candidate_core["recolor_map"],
                        "selected_object": obj,
                        "object": obj,
                        "predicted": candidate_core["predicted"],
                        "score": candidate_core["score"],
                        "exact": candidate_core["exact"],
                    }

                    if candidate["score"] > best_score:
                        best_score = candidate["score"]
                        best = candidate

    return best