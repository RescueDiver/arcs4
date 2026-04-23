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

GROUP_MODES = [
    "all_non_full_grid",
    "all_not_touches_border",
    "all_touches_border",
    "all_objects_except_smallest",
]

PLACEMENT_MODES = [
    "overlay_anchor",
    "vertical_stack",
    "horizontal_stack",
    "relative_rows",
    "relative_cols",
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


def make_blank_grid(h, w, fill=0):
    return [[fill for _ in range(w)] for _ in range(h)]


def overlay_nonzero(base, layer):
    h = len(base)
    w = len(base[0]) if h else 0
    out = [row[:] for row in base]

    for r in range(h):
        for c in range(w):
            if layer[r][c] != 0:
                out[r][c] = layer[r][c]

    return out


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


def sort_objects_reading_order(objects):
    return sorted(
        objects,
        key=lambda o: (
            o["bbox"]["min_r"],
            o["bbox"]["min_c"],
            o["bbox"]["max_r"],
            o["bbox"]["max_c"],
        ),
    )


def select_object_group(objects, mode, rows, cols):
    if not objects:
        return []

    sorted_objs = sort_objects_reading_order(objects)

    if mode == "all_non_full_grid":
        return [o for o in sorted_objs if not is_full_grid_object(o, rows, cols)]

    if mode == "all_not_touches_border":
        return [o for o in sorted_objs if not touches_border(o, rows, cols)]

    if mode == "all_touches_border":
        return [o for o in sorted_objs if touches_border(o, rows, cols)]

    if mode == "all_objects_except_smallest":
        if len(sorted_objs) <= 1:
            return sorted_objs[:]
        min_area = min(o["area"] for o in sorted_objs)
        return [o for o in sorted_objs if o["area"] > min_area]

    return []


def stamp_patch(canvas, patch, start_r, start_c):
    out_h = len(canvas)
    out_w = len(canvas[0]) if out_h else 0
    ph = len(patch)
    pw = len(patch[0]) if ph else 0

    if ph == 0 or pw == 0:
        return canvas

    out = [row[:] for row in canvas]

    for r in range(ph):
        for c in range(pw):
            rr = start_r + r
            cc = start_c + c
            if 0 <= rr < out_h and 0 <= cc < out_w and patch[r][c] != 0:
                out[rr][cc] = patch[r][c]

    return out


def build_overlay_anchor_canvas(group, out_h, out_w, transform, v_align, h_align):
    canvas = make_blank_grid(out_h, out_w, fill=0)
    used_any = False

    for obj in group:
        patch = apply_transform(obj["patch"], transform)
        ph = len(patch)
        pw = len(patch[0]) if ph else 0
        if ph == 0 or pw == 0:
            continue

        start_r = get_anchor_start(out_h, ph, v_align)
        start_c = get_anchor_start(out_w, pw, h_align)
        if start_r is None or start_c is None:
            continue

        canvas = stamp_patch(canvas, patch, start_r, start_c)
        used_any = True

    return canvas if used_any else None


def build_vertical_stack_canvas(group, out_h, out_w, transform, h_align):
    canvas = make_blank_grid(out_h, out_w, fill=0)
    current_r = 0
    used_any = False

    for obj in group:
        patch = apply_transform(obj["patch"], transform)
        ph = len(patch)
        pw = len(patch[0]) if ph else 0
        if ph == 0 or pw == 0 or ph > out_h or pw > out_w:
            continue

        start_c = get_anchor_start(out_w, pw, h_align)
        if start_c is None:
            continue

        if current_r + ph > out_h:
            break

        canvas = stamp_patch(canvas, patch, current_r, start_c)
        current_r += ph
        used_any = True

    return canvas if used_any else None


def build_horizontal_stack_canvas(group, out_h, out_w, transform, v_align):
    canvas = make_blank_grid(out_h, out_w, fill=0)
    current_c = 0
    used_any = False

    for obj in group:
        patch = apply_transform(obj["patch"], transform)
        ph = len(patch)
        pw = len(patch[0]) if ph else 0
        if ph == 0 or pw == 0 or ph > out_h or pw > out_w:
            continue

        start_r = get_anchor_start(out_h, ph, v_align)
        if start_r is None:
            continue

        if current_c + pw > out_w:
            break

        canvas = stamp_patch(canvas, patch, start_r, current_c)
        current_c += pw
        used_any = True

    return canvas if used_any else None


def normalize_positions(values, out_size, patch_sizes):
    if not values:
        return []

    vmin = min(values)
    vmax = max(values)

    if vmax == vmin:
        # all same position
        return [0 for _ in values]

    usable = max(0, out_size - max(patch_sizes))
    normed = []

    for v in values:
        ratio = (v - vmin) / (vmax - vmin)
        pos = int(round(ratio * usable))
        normed.append(pos)

    return normed


def build_relative_rows_canvas(group, out_h, out_w, transform, h_align):
    patches = []
    row_positions = []

    for obj in group:
        patch = apply_transform(obj["patch"], transform)
        ph = len(patch)
        pw = len(patch[0]) if ph else 0
        if ph == 0 or pw == 0 or ph > out_h or pw > out_w:
            continue
        patches.append((obj, patch, ph, pw))
        row_positions.append(obj["bbox"]["min_r"])

    if not patches:
        return None

    starts_r = normalize_positions(row_positions, out_h, [ph for _, _, ph, _ in patches])
    canvas = make_blank_grid(out_h, out_w, fill=0)

    for (obj, patch, ph, pw), start_r in zip(patches, starts_r):
        start_c = get_anchor_start(out_w, pw, h_align)
        if start_c is None:
            continue
        canvas = stamp_patch(canvas, patch, start_r, start_c)

    return canvas


def build_relative_cols_canvas(group, out_h, out_w, transform, v_align):
    patches = []
    col_positions = []

    for obj in group:
        patch = apply_transform(obj["patch"], transform)
        ph = len(patch)
        pw = len(patch[0]) if ph else 0
        if ph == 0 or pw == 0 or ph > out_h or pw > out_w:
            continue
        patches.append((obj, patch, ph, pw))
        col_positions.append(obj["bbox"]["min_c"])

    if not patches:
        return None

    starts_c = normalize_positions(col_positions, out_w, [pw for _, _, _, pw in patches])
    canvas = make_blank_grid(out_h, out_w, fill=0)

    for (obj, patch, ph, pw), start_c in zip(patches, starts_c):
        start_r = get_anchor_start(out_h, ph, v_align)
        if start_r is None:
            continue
        canvas = stamp_patch(canvas, patch, start_r, start_c)

    return canvas


def build_canvas_for_mode(group, output_grid, transform, placement_mode, v_align, h_align):
    out_h = len(output_grid)
    out_w = len(output_grid[0]) if out_h else 0

    if placement_mode == "overlay_anchor":
        return build_overlay_anchor_canvas(group, out_h, out_w, transform, v_align, h_align)

    if placement_mode == "vertical_stack":
        return build_vertical_stack_canvas(group, out_h, out_w, transform, h_align)

    if placement_mode == "horizontal_stack":
        return build_horizontal_stack_canvas(group, out_h, out_w, transform, v_align)

    if placement_mode == "relative_rows":
        return build_relative_rows_canvas(group, out_h, out_w, transform, h_align)

    if placement_mode == "relative_cols":
        return build_relative_cols_canvas(group, out_h, out_w, transform, v_align)

    return None


def try_group_projection(group, output_grid, transform, placement_mode, v_align, h_align):
    canvas = build_canvas_for_mode(
        group=group,
        output_grid=output_grid,
        transform=transform,
        placement_mode=placement_mode,
        v_align=v_align,
        h_align=h_align,
    )
    if canvas is None:
        return None

    recolor_map = None
    predicted = canvas
    exact = predicted == output_grid
    score = score_prediction(predicted, output_grid)

    if not exact and same_shape_mask(predicted, output_grid):
        maybe_map = infer_recolor_map(predicted, output_grid)
        if maybe_map is not None:
            recolored = apply_recolor_map(predicted, maybe_map)
            recolored_score = score_prediction(recolored, output_grid)
            if recolored_score >= score:
                predicted = recolored
                recolor_map = maybe_map
                score = recolored_score
                exact = predicted == output_grid

    return {
        "predicted": predicted,
        "transform": transform,
        "placement_mode": placement_mode,
        "v_align": v_align,
        "h_align": h_align,
        "recolor_map": recolor_map,
        "score": score,
        "exact": exact,
    }


def solve_pair_multi_object_projection_rule(input_grid, output_grid):
    objects = find_objects(input_grid)
    if not objects:
        return None

    rows = len(input_grid)
    cols = len(input_grid[0]) if rows else 0

    best = None
    best_score = -10**9

    for mode in GROUP_MODES:
        group = select_object_group(objects, mode, rows, cols)
        if not group:
            continue

        for obj in group:
            debug_selected_object(f"MULTI PROJECTION OBJECT ({mode})", obj)

        for transform in TRANSFORMS:
            for placement_mode in PLACEMENT_MODES:
                for v_align in V_ALIGNS:
                    for h_align in H_ALIGNS:
                        candidate_core = try_group_projection(
                            group=group,
                            output_grid=output_grid,
                            transform=transform,
                            placement_mode=placement_mode,
                            v_align=v_align,
                            h_align=h_align,
                        )
                        if candidate_core is None:
                            continue

                        candidate = {
                            "strategy": "multi_object_projection_rule",
                            "mode": mode,
                            "placement_mode": placement_mode,
                            "transform": candidate_core["transform"],
                            "v_align": candidate_core["v_align"],
                            "h_align": candidate_core["h_align"],
                            "recolor_map": candidate_core["recolor_map"],
                            "group_size": len(group),
                            "predicted": candidate_core["predicted"],
                            "score": candidate_core["score"],
                            "exact": candidate_core["exact"],
                        }

                        if candidate["score"] > best_score:
                            best_score = candidate["score"]
                            best = candidate

    return best