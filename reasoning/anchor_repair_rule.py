# reasoning/anchor_repair_rule.py
"""
Anchor repair / anchor marker merge rule.

Learns tasks where:
    - one color appears in input and disappears in output = anchor
    - the remaining colors are tried as background/structure
    - anchor marks a joining face
    - one anchor object can act as base
    - the other anchor object can be inserted at the base anchor face
    - anchor disappears / becomes absorbed into structure
    - result is cropped

No task ids. No hardcoding.
"""


# ============================================================
# BASIC HELPERS
# ============================================================

def grid_shape(grid):
    if grid is None:
        return 0, 0

    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def make_grid(h, w, fill):
    return [[fill for _ in range(w)] for _ in range(h)]


def copy_grid(grid):
    return [row[:] for row in grid]


def colors_in_grid(grid):
    colors = set()

    for row in grid:
        for value in row:
            colors.add(value)

    return colors


def bounding_box_of_cells(cells):
    if not cells:
        return None

    rows = [r for r, c in cells]
    cols = [c for r, c in cells]

    return {
        "top": min(rows),
        "bottom": max(rows),
        "left": min(cols),
        "right": max(cols),
        "height": max(rows) - min(rows) + 1,
        "width": max(cols) - min(cols) + 1,
    }


def crop_grid(grid, box):
    if box is None:
        return None

    return [
        row[box["left"]:box["right"] + 1]
        for row in grid[box["top"]:box["bottom"] + 1]
    ]


def crop_to_non_background(grid, background):
    h, w = grid_shape(grid)

    cells = []

    for r in range(h):
        for c in range(w):
            if grid[r][c] != background:
                cells.append((r, c))

    box = bounding_box_of_cells(cells)

    if box is None:
        return None

    return crop_grid(grid, box)


def score_prediction_simple(predicted, expected):
    if predicted is None or expected is None:
        return -100000

    ph, pw = grid_shape(predicted)
    eh, ew = grid_shape(expected)

    score = 0

    for r in range(min(ph, eh)):
        for c in range(min(pw, ew)):
            if predicted[r][c] == expected[r][c]:
                score += 1

    score -= abs(ph - eh) * 50
    score -= abs(pw - ew) * 50

    if predicted == expected:
        score += 1000000

    return score


# ============================================================
# COMPONENT HELPERS
# ============================================================

def find_components_by_color_set(grid, allowed_colors):
    h, w = grid_shape(grid)

    seen = set()
    components = []

    for r in range(h):
        for c in range(w):
            if grid[r][c] not in allowed_colors:
                continue

            if (r, c) in seen:
                continue

            stack = [(r, c)]
            seen.add((r, c))
            cells = []

            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = rr + dr
                    nc = cc + dc

                    if not (0 <= nr < h and 0 <= nc < w):
                        continue

                    if (nr, nc) in seen:
                        continue

                    if grid[nr][nc] not in allowed_colors:
                        continue

                    seen.add((nr, nc))
                    stack.append((nr, nc))

            counts = {}

            for rr, cc in cells:
                value = grid[rr][cc]
                counts[value] = counts.get(value, 0) + 1

            components.append({
                "cells": cells,
                "bbox": bounding_box_of_cells(cells),
                "cell_count": len(cells),
                "color_counts": counts,
                "colors": sorted(counts.keys()),
            })

    components.sort(
        key=lambda obj: (
            obj["bbox"]["top"],
            obj["bbox"]["left"],
            obj["cell_count"],
        )
    )

    return components


def component_has_color(component, color):
    return color in component["color_counts"]


def anchor_cells_in_component(grid, component, anchor_color):
    cells = []

    for r, c in component["cells"]:
        if grid[r][c] == anchor_color:
            cells.append((r, c))

    return cells


def structure_cells_in_component(grid, component, structure_color):
    cells = []

    for r, c in component["cells"]:
        if grid[r][c] == structure_color:
            cells.append((r, c))

    return cells


def center_of_box(box):
    return (
        box["top"] + box["height"] / 2,
        box["left"] + box["width"] / 2,
    )


def anchor_reference_points(anchor_cells):
    box = bounding_box_of_cells(anchor_cells)

    if box is None:
        return {}

    center_r = box["top"] + box["height"] // 2
    center_c = box["left"] + box["width"] // 2

    return {
        "top_left": (box["top"], box["left"]),
        "top_center": (box["top"], center_c),
        "top_right": (box["top"], box["right"]),
        "center_left": (center_r, box["left"]),
        "center": (center_r, center_c),
        "center_right": (center_r, box["right"]),
        "bottom_left": (box["bottom"], box["left"]),
        "bottom_center": (box["bottom"], center_c),
        "bottom_right": (box["bottom"], box["right"]),
    }


def local_anchor_reference_points(component, anchor_cells):
    component_box = component["bbox"]
    points = anchor_reference_points(anchor_cells)

    local = {}

    for name, point in points.items():
        r, c = point
        local[name] = (
            r - component_box["top"],
            c - component_box["left"],
        )

    return local


def select_base_patch(anchor_components, base_selector):
    if len(anchor_components) != 2:
        return None, None

    a = anchor_components[0]
    b = anchor_components[1]

    if base_selector == "first":
        return a, b

    if base_selector == "second":
        return b, a

    if base_selector == "largest":
        ordered = sorted(
            anchor_components,
            key=lambda obj: obj["cell_count"],
            reverse=True,
        )
        return ordered[0], ordered[1]

    if base_selector == "smallest":
        ordered = sorted(
            anchor_components,
            key=lambda obj: obj["cell_count"],
        )
        return ordered[0], ordered[1]

    if base_selector == "topmost":
        ordered = sorted(
            anchor_components,
            key=lambda obj: obj["bbox"]["top"],
        )
        return ordered[0], ordered[1]

    if base_selector == "bottommost":
        ordered = sorted(
            anchor_components,
            key=lambda obj: obj["bbox"]["top"],
            reverse=True,
        )
        return ordered[0], ordered[1]

    if base_selector == "leftmost":
        ordered = sorted(
            anchor_components,
            key=lambda obj: obj["bbox"]["left"],
        )
        return ordered[0], ordered[1]

    if base_selector == "rightmost":
        ordered = sorted(
            anchor_components,
            key=lambda obj: obj["bbox"]["left"],
            reverse=True,
        )
        return ordered[0], ordered[1]

    return None, None


def bbox_center(box):
    return (
        box["top"] + (box["height"] - 1) / 2,
        box["left"] + (box["width"] - 1) / 2,
    )


def distance_between_boxes(box_a, box_b):
    ar, ac = bbox_center(box_a)
    br, bc = bbox_center(box_b)

    return abs(ar - br) + abs(ac - bc)


def assign_structure_satellites_to_anchor_components(
    input_grid,
    structure_color,
    anchor_color,
    anchor_components,
):
    """
    Find structure-only components and attach each one to the nearest
    anchor-bearing component.

    This handles pair 1:
        separate yellow 4 4 piece belongs to the small anchor patch.
    """
    structure_components = find_components_by_color_set(
        input_grid,
        {structure_color},
    )

    satellites_by_anchor_id = {}

    for i in range(len(anchor_components)):
        satellites_by_anchor_id[i] = []

    for component in structure_components:
        # Ignore structure components that overlap an anchor-bearing component.
        overlaps_anchor_component = False

        component_cells = set(component["cells"])

        for anchor_component in anchor_components:
            anchor_cells = set(anchor_component["cells"])

            if component_cells & anchor_cells:
                overlaps_anchor_component = True
                break

        if overlaps_anchor_component:
            continue

        best_index = None
        best_distance = None

        for i, anchor_component in enumerate(anchor_components):
            distance = distance_between_boxes(
                component["bbox"],
                anchor_component["bbox"],
            )

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = i

        if best_index is not None:
            satellites_by_anchor_id[best_index].append(component)

    return satellites_by_anchor_id


def component_index_in_list(target_component, components):
    for i, component in enumerate(components):
        if component is target_component:
            return i

    return None


def draw_satellite_components(
    cells,
    input_grid,
    satellites,
    anchor_component,
    anchor_component_top,
    anchor_component_left,
    structure_color,
):
    """
    Satellites keep their original offset relative to the anchor-bearing
    component they belong to.
    """
    anchor_box = anchor_component["bbox"]

    for satellite in satellites:
        for r, c in satellite["cells"]:
            if input_grid[r][c] != structure_color:
                continue

            local_r = anchor_component_top + (r - anchor_box["top"])
            local_c = anchor_component_left + (c - anchor_box["left"])

            cells.append((local_r, local_c, structure_color))


# ============================================================
# COLOR LEARNING
# ============================================================

def infer_colors(train_pairs):
    """
    Learn only stable color facts.

    Anchor:
        color that appears in every input and disappears from every output.

    Remaining colors are tried both ways:
        background / structure
        structure / background
    """
    if not train_pairs:
        return None

    possible_anchor_colors = None
    remaining_colors = set()

    for pair in train_pairs:
        inp = pair["input"]
        out = pair["output"]

        input_colors = colors_in_grid(inp)
        output_colors = colors_in_grid(out)

        disappeared = input_colors - output_colors

        if not disappeared:
            return None

        if possible_anchor_colors is None:
            possible_anchor_colors = set(disappeared)
        else:
            possible_anchor_colors &= disappeared

        remaining_colors |= input_colors
        remaining_colors |= output_colors

    if not possible_anchor_colors:
        return None

    anchor_color = sorted(possible_anchor_colors)[0]

    remaining_colors = sorted(
        color for color in remaining_colors
        if color != anchor_color
    )

    if len(remaining_colors) < 2:
        return None

    return {
        "anchor_color": anchor_color,
        "remaining_colors": remaining_colors,
    }


# ============================================================
# DRAWING HELPERS
# ============================================================

def draw_component_into_canvas(
    canvas,
    input_grid,
    component,
    top,
    left,
    structure_color,
    anchor_color,
    background_color,
    draw_anchor_as_structure,
    draw_structure,
):
    h, w = grid_shape(canvas)
    box = component["bbox"]

    for r, c in component["cells"]:
        value = input_grid[r][c]

        rr = top + (r - box["top"])
        cc = left + (c - box["left"])

        if not (0 <= rr < h and 0 <= cc < w):
            continue

        if value == structure_color and draw_structure:
            canvas[rr][cc] = structure_color

        elif value == anchor_color and draw_anchor_as_structure:
            canvas[rr][cc] = structure_color


def draw_patch_body_cells(
    canvas,
    input_grid,
    component,
    top,
    left,
    structure_color,
    anchor_color=None,
    draw_anchor_as_structure=False,
):
    h, w = grid_shape(canvas)
    box = component["bbox"]

    for r, c in component["cells"]:
        value = input_grid[r][c]

        if value == structure_color:
            should_draw = True
        elif draw_anchor_as_structure and value == anchor_color:
            should_draw = True
        else:
            should_draw = False

        if not should_draw:
            continue

        rr = top + (r - box["top"])
        cc = left + (c - box["left"])

        if 0 <= rr < h and 0 <= cc < w:
            canvas[rr][cc] = structure_color


def expand_and_draw(base_items, background_color):
    """
    base_items:
        list of dicts:
            cells: [(r,c,color)]
    """
    all_cells = []

    for item in base_items:
        all_cells.extend(item["cells"])

    if not all_cells:
        return None

    box = bounding_box_of_cells([
        (r, c)
        for r, c, color in all_cells
    ])

    if box is None:
        return None

    out = make_grid(
        box["height"],
        box["width"],
        background_color,
    )

    row_shift = -box["top"]
    col_shift = -box["left"]

    for r, c, color in all_cells:
        out[r + row_shift][c + col_shift] = color

    return crop_to_non_background(
        out,
        background_color,
    )


# ============================================================
# CANDIDATE BUILDERS
# ============================================================

def build_replace_anchor_crop(input_grid, anchor_color, structure_color, background_color):
    out = copy_grid(input_grid)

    h, w = grid_shape(out)

    for r in range(h):
        for c in range(w):
            if out[r][c] == anchor_color:
                out[r][c] = structure_color

    return crop_to_non_background(out, background_color)


def build_anchor_mask_relative_merge(
    input_grid,
    anchor_color,
    structure_color,
    background_color,
    anchor_reference_name,
):
    components = find_components_by_color_set(
        input_grid,
        {structure_color, anchor_color},
    )

    anchor_components = [
        component
        for component in components
        if component_has_color(component, anchor_color)
    ]

    if len(anchor_components) < 2:
        return None

    merged_relative_cells = set()

    for component in anchor_components:
        anchors = anchor_cells_in_component(
            input_grid,
            component,
            anchor_color,
        )

        points = anchor_reference_points(anchors)

        if anchor_reference_name not in points:
            return None

        ref_r, ref_c = points[anchor_reference_name]

        for r, c in component["cells"]:
            value = input_grid[r][c]

            if value not in {structure_color, anchor_color}:
                continue

            merged_relative_cells.add((
                r - ref_r,
                c - ref_c,
            ))

    if not merged_relative_cells:
        return None

    box = bounding_box_of_cells(list(merged_relative_cells))

    if box is None:
        return None

    out = make_grid(
        box["height"],
        box["width"],
        background_color,
    )

    row_shift = -box["top"]
    col_shift = -box["left"]

    for r, c in merged_relative_cells:
        out[r + row_shift][c + col_shift] = structure_color

    return out


def crop_component_structure(grid, component, structure_color):
    cells = structure_cells_in_component(
        grid,
        component,
        structure_color,
    )

    box = bounding_box_of_cells(cells)

    if box is None:
        return None

    local_cells = []

    for r, c in cells:
        local_cells.append((
            r - box["top"],
            c - box["left"],
        ))

    return {
        "height": box["height"],
        "width": box["width"],
        "cells": local_cells,
        "bbox": box,
    }


def draw_piece(canvas, piece, top, left, structure_color):
    h, w = grid_shape(canvas)

    for r, c in piece["cells"]:
        rr = top + r
        cc = left + c

        if 0 <= rr < h and 0 <= cc < w:
            canvas[rr][cc] = structure_color


def build_vertical_edge_join(
    input_grid,
    anchor_color,
    structure_color,
    background_color,
    horizontal_align,
):
    components = find_components_by_color_set(
        input_grid,
        {structure_color, anchor_color},
    )

    anchor_components = [
        component
        for component in components
        if component_has_color(component, anchor_color)
    ]

    if len(anchor_components) != 2:
        return None

    records = []

    for component in anchor_components:
        anchors = anchor_cells_in_component(
            input_grid,
            component,
            anchor_color,
        )

        anchor_box = bounding_box_of_cells(anchors)

        piece = crop_component_structure(
            input_grid,
            component,
            structure_color,
        )

        if anchor_box is None or piece is None:
            return None

        records.append({
            "anchor_box": anchor_box,
            "piece": piece,
        })

    records.sort(
        key=lambda item: center_of_box(item["anchor_box"])[0]
    )

    top_piece = records[0]["piece"]
    bottom_piece = records[1]["piece"]

    out_h = top_piece["height"] + bottom_piece["height"]
    out_w = max(top_piece["width"], bottom_piece["width"])

    canvas = make_grid(out_h, out_w, background_color)

    if horizontal_align == "left":
        top_left = 0
        bottom_left = 0
    elif horizontal_align == "right":
        top_left = out_w - top_piece["width"]
        bottom_left = out_w - bottom_piece["width"]
    else:
        top_left = (out_w - top_piece["width"]) // 2
        bottom_left = (out_w - bottom_piece["width"]) // 2

    draw_piece(canvas, top_piece, 0, top_left, structure_color)
    draw_piece(canvas, bottom_piece, top_piece["height"], bottom_left, structure_color)

    return crop_to_non_background(canvas, background_color)


def build_horizontal_edge_join(
    input_grid,
    anchor_color,
    structure_color,
    background_color,
    vertical_align,
):
    components = find_components_by_color_set(
        input_grid,
        {structure_color, anchor_color},
    )

    anchor_components = [
        component
        for component in components
        if component_has_color(component, anchor_color)
    ]

    if len(anchor_components) != 2:
        return None

    records = []

    for component in anchor_components:
        anchors = anchor_cells_in_component(
            input_grid,
            component,
            anchor_color,
        )

        anchor_box = bounding_box_of_cells(anchors)

        piece = crop_component_structure(
            input_grid,
            component,
            structure_color,
        )

        if anchor_box is None or piece is None:
            return None

        records.append({
            "anchor_box": anchor_box,
            "piece": piece,
        })

    records.sort(
        key=lambda item: center_of_box(item["anchor_box"])[1]
    )

    left_piece = records[0]["piece"]
    right_piece = records[1]["piece"]

    out_h = max(left_piece["height"], right_piece["height"])
    out_w = left_piece["width"] + right_piece["width"]

    canvas = make_grid(out_h, out_w, background_color)

    if vertical_align == "top":
        left_top = 0
        right_top = 0
    elif vertical_align == "bottom":
        left_top = out_h - left_piece["height"]
        right_top = out_h - right_piece["height"]
    else:
        left_top = (out_h - left_piece["height"]) // 2
        right_top = (out_h - right_piece["height"]) // 2

    draw_piece(canvas, left_piece, left_top, 0, structure_color)
    draw_piece(canvas, right_piece, right_top, left_piece["width"], structure_color)

    return crop_to_non_background(canvas, background_color)


def build_anchor_edge_join_compress(
    input_grid,
    anchor_color,
    structure_color,
    background_color,
    join_axis,
    align_mode,
):
    if join_axis == "vertical":
        return build_vertical_edge_join(
            input_grid,
            anchor_color,
            structure_color,
            background_color,
            horizontal_align=align_mode,
        )

    if join_axis == "horizontal":
        return build_horizontal_edge_join(
            input_grid,
            anchor_color,
            structure_color,
            background_color,
            vertical_align=align_mode,
        )

    return None


def shift_from_name(shift_name):
    shifts = {
        "same": (0, 0),
        "up1": (-1, 0),
        "down1": (1, 0),
        "left1": (0, -1),
        "right1": (0, 1),
        "up2": (-2, 0),
        "down2": (2, 0),
        "left2": (0, -2),
        "right2": (0, 2),
    }

    return shifts.get(shift_name, (0, 0))


def build_anchor_face_insert(
    input_grid,
    anchor_color,
    structure_color,
    background_color,
    base_selector,
    base_anchor_ref,
    patch_anchor_ref,
    shift_name,
    canvas_mode,
    patch_anchor_mode,
):
    """
    New hypothesis:

        base object stays as the main canvas
        base anchor cells become structure
        patch anchor cells may be ignored or absorbed
        patch structure cells are inserted near the base anchor face
        structure-only satellites move with their nearest anchor component
    """
    components = find_components_by_color_set(
        input_grid,
        {structure_color, anchor_color},
    )

    anchor_components = [
        component
        for component in components
        if component_has_color(component, anchor_color)
    ]

    if len(anchor_components) != 2:
        return None

    satellites_by_anchor_id = assign_structure_satellites_to_anchor_components(
        input_grid=input_grid,
        structure_color=structure_color,
        anchor_color=anchor_color,
        anchor_components=anchor_components,
    )

    base, patch = select_base_patch(
        anchor_components,
        base_selector,
    )

    if base is None or patch is None:
        return None

    base_index = component_index_in_list(base, anchor_components)
    patch_index = component_index_in_list(patch, anchor_components)

    if base_index is None or patch_index is None:
        return None

    base_box = base["bbox"]
    patch_box = patch["bbox"]

    base_anchor_cells = anchor_cells_in_component(
        input_grid,
        base,
        anchor_color,
    )

    patch_anchor_cells = anchor_cells_in_component(
        input_grid,
        patch,
        anchor_color,
    )

    base_points = local_anchor_reference_points(
        base,
        base_anchor_cells,
    )

    patch_points = local_anchor_reference_points(
        patch,
        patch_anchor_cells,
    )

    if base_anchor_ref not in base_points:
        return None

    if patch_anchor_ref not in patch_points:
        return None

    base_ref_r, base_ref_c = base_points[base_anchor_ref]
    patch_ref_r, patch_ref_c = patch_points[patch_anchor_ref]

    shift_r, shift_c = shift_from_name(shift_name)

    patch_top = base_ref_r + shift_r - patch_ref_r
    patch_left = base_ref_c + shift_c - patch_ref_c

    if canvas_mode == "fixed_base":
        canvas = make_grid(
            base_box["height"],
            base_box["width"],
            background_color,
        )

        draw_component_into_canvas(
            canvas=canvas,
            input_grid=input_grid,
            component=base,
            top=0,
            left=0,
            structure_color=structure_color,
            anchor_color=anchor_color,
            background_color=background_color,
            draw_anchor_as_structure=True,
            draw_structure=True,
        )

        draw_patch_body_cells(
            canvas=canvas,
            input_grid=input_grid,
            component=patch,
            top=patch_top,
            left=patch_left,
            structure_color=structure_color,
            anchor_color=anchor_color,
            draw_anchor_as_structure=(patch_anchor_mode == "draw_as_structure"),
        )

        # Draw base satellites.
        base_satellite_cells = []
        draw_satellite_components(
            cells=base_satellite_cells,
            input_grid=input_grid,
            satellites=satellites_by_anchor_id.get(base_index, []),
            anchor_component=base,
            anchor_component_top=0,
            anchor_component_left=0,
            structure_color=structure_color,
        )

        for r, c, color in base_satellite_cells:
            if 0 <= r < len(canvas) and 0 <= c < len(canvas[0]):
                canvas[r][c] = color

        # Draw patch satellites.
        patch_satellite_cells = []
        draw_satellite_components(
            cells=patch_satellite_cells,
            input_grid=input_grid,
            satellites=satellites_by_anchor_id.get(patch_index, []),
            anchor_component=patch,
            anchor_component_top=patch_top,
            anchor_component_left=patch_left,
            structure_color=structure_color,
        )

        for r, c, color in patch_satellite_cells:
            if 0 <= r < len(canvas) and 0 <= c < len(canvas[0]):
                canvas[r][c] = color

        return crop_to_non_background(
            canvas,
            background_color,
        )

    if canvas_mode == "expand":
        cells = []

        # Base structure + base anchor as structure.
        for r, c in base["cells"]:
            value = input_grid[r][c]

            local_r = r - base_box["top"]
            local_c = c - base_box["left"]

            if value == structure_color or value == anchor_color:
                cells.append((local_r, local_c, structure_color))

        # Base satellites.
        draw_satellite_components(
            cells=cells,
            input_grid=input_grid,
            satellites=satellites_by_anchor_id.get(base_index, []),
            anchor_component=base,
            anchor_component_top=0,
            anchor_component_left=0,
            structure_color=structure_color,
        )

        # Patch structure, optionally patch anchor as structure.
        for r, c in patch["cells"]:
            value = input_grid[r][c]

            if value == structure_color:
                should_draw = True
            elif patch_anchor_mode == "draw_as_structure" and value == anchor_color:
                should_draw = True
            else:
                should_draw = False

            if not should_draw:
                continue

            local_r = patch_top + (r - patch_box["top"])
            local_c = patch_left + (c - patch_box["left"])

            cells.append((local_r, local_c, structure_color))

        # Patch satellites.
        draw_satellite_components(
            cells=cells,
            input_grid=input_grid,
            satellites=satellites_by_anchor_id.get(patch_index, []),
            anchor_component=patch,
            anchor_component_top=patch_top,
            anchor_component_left=patch_left,
            structure_color=structure_color,
        )

        return expand_and_draw(
            [{"cells": cells}],
            background_color,
        )

    return None


def build_from_rule(rule, input_grid):
    if rule is None:
        return None

    anchor_color = rule.get("anchor_color")
    structure_color = rule.get("structure_color")
    background_color = rule.get("background_color")
    rule_type = rule.get("rule_type")

    if rule_type == "replace_anchor_crop":
        return build_replace_anchor_crop(
            input_grid,
            anchor_color,
            structure_color,
            background_color,
        )

    if rule_type == "anchor_face_insert":
        return build_anchor_face_insert(
            input_grid=input_grid,
            anchor_color=anchor_color,
            structure_color=structure_color,
            background_color=background_color,
            base_selector=rule.get("base_selector"),
            base_anchor_ref=rule.get("base_anchor_ref"),
            patch_anchor_ref=rule.get("patch_anchor_ref"),
            shift_name=rule.get("shift_name"),
            canvas_mode=rule.get("canvas_mode"),
            patch_anchor_mode=rule.get("patch_anchor_mode"),
        )

    if rule_type == "anchor_edge_join_compress":
        return build_anchor_edge_join_compress(
            input_grid=input_grid,
            anchor_color=anchor_color,
            structure_color=structure_color,
            background_color=background_color,
            join_axis=rule.get("join_axis"),
            align_mode=rule.get("align_mode"),
        )

    if rule_type == "anchor_face_insert":
        return build_anchor_face_insert(
            input_grid=input_grid,
            anchor_color=anchor_color,
            structure_color=structure_color,
            background_color=background_color,
            base_selector=rule.get("base_selector"),
            base_anchor_ref=rule.get("base_anchor_ref"),
            patch_anchor_ref=rule.get("patch_anchor_ref"),
            shift_name=rule.get("shift_name"),
            canvas_mode=rule.get("canvas_mode"),
        )

    return None


# ============================================================
# LEARNING
# ============================================================

def score_rule_on_train(rule, train_pairs):
    exact_count = 0
    total_score = 0
    examples = []

    for pair_index, pair in enumerate(train_pairs):
        predicted = build_from_rule(
            rule,
            pair["input"],
        )

        expected = pair["output"]
        exact = predicted == expected
        score = score_prediction_simple(predicted, expected)

        if exact:
            exact_count += 1

        total_score += score

        examples.append({
            "pair_index": pair_index,
            "input_shape": grid_shape(pair["input"]),
            "output_shape": grid_shape(expected),
            "predicted_shape": grid_shape(predicted),
            "exact": exact,
            "score": score,
        })

    return {
        "exact_count": exact_count,
        "total_score": total_score,
        "examples": examples,
    }


def generate_candidate_rules(base_rule):
    candidates = []

    rule = dict(base_rule)
    rule["rule_type"] = "replace_anchor_crop"
    candidates.append(rule)

    for anchor_reference_name in [
        "top_left",
        "top_center",
        "top_right",
        "center_left",
        "center",
        "center_right",
        "bottom_left",
        "bottom_center",
        "bottom_right",
    ]:
        rule = dict(base_rule)
        rule["rule_type"] = "anchor_mask_relative_merge"
        rule["anchor_reference_name"] = anchor_reference_name
        candidates.append(rule)

    for join_axis in ["vertical", "horizontal"]:
        if join_axis == "vertical":
            align_modes = ["left", "center", "right"]
        else:
            align_modes = ["top", "center", "bottom"]

        for align_mode in align_modes:
            rule = dict(base_rule)
            rule["rule_type"] = "anchor_edge_join_compress"
            rule["join_axis"] = join_axis
            rule["align_mode"] = align_mode
            candidates.append(rule)

    refs = [
        "top_left",
        "top_center",
        "top_right",
        "center_left",
        "center",
        "center_right",
        "bottom_left",
        "bottom_center",
        "bottom_right",
    ]

    base_selectors = [
        "largest",
        "smallest",
        "topmost",
        "bottommost",
        "leftmost",
        "rightmost",
        "first",
        "second",
    ]

    shift_names = [
        "same",
        "up1",
        "down1",
        "left1",
        "right1",
        "up2",
        "down2",
        "left2",
        "right2",
    ]

    canvas_modes = [
        "fixed_base",
        "expand",
    ]
    patch_anchor_modes = [
        "ignore",
        "draw_as_structure",
    ]
    for base_selector in base_selectors:
        for base_anchor_ref in refs:
            for patch_anchor_ref in refs:
                for shift_name in shift_names:
                    for canvas_mode in canvas_modes:
                        for patch_anchor_mode in patch_anchor_modes:
                            rule = dict(base_rule)
                            rule["rule_type"] = "anchor_face_insert"
                            rule["base_selector"] = base_selector
                            rule["base_anchor_ref"] = base_anchor_ref
                            rule["patch_anchor_ref"] = patch_anchor_ref
                            rule["shift_name"] = shift_name
                            rule["canvas_mode"] = canvas_mode
                            rule["patch_anchor_mode"] = patch_anchor_mode
                            candidates.append(rule)

    return candidates


def learn_anchor_repair_rule(train_pairs):
    if not train_pairs:
        return None

    learned = infer_colors(train_pairs)


    if learned is None:
        return None

    anchor_color = learned["anchor_color"]
    remaining_colors = learned["remaining_colors"]

    candidates = []

    for structure_color in remaining_colors:
        for background_color in remaining_colors:
            if structure_color == background_color:
                continue

            base_rule = {
                "family": "anchor_repair_rule",
                "anchor_color": anchor_color,
                "structure_color": structure_color,
                "background_color": background_color,
            }

            candidates.extend(
                generate_candidate_rules(base_rule)
            )

    scored = []

    for rule in candidates:
        score_info = score_rule_on_train(
            rule,
            train_pairs,
        )

        item = dict(rule)
        item["exact_count"] = score_info["exact_count"]
        item["total_score"] = score_info["total_score"]
        item["pair_count"] = len(train_pairs)
        item["examples"] = score_info["examples"]

        scored.append(item)

    scored.sort(
        key=lambda item: (
            item.get("exact_count", 0),
            item.get("total_score", -1000000),
        ),
        reverse=True,
    )

    best = scored[0]
    best["debug_candidate_count"] = len(scored)
    best["debug_top_candidates"] = scored[:5]



    return best


# ============================================================
# PUBLIC API
# ============================================================

def predict_anchor_repair_for_pair(rule, input_grid):
    return build_from_rule(
        rule,
        input_grid,
    )


def solve_pair_anchor_repair_rule(input_grid, output_grid):
    pair = {
        "input": input_grid,
        "output": output_grid,
    }

    rule = learn_anchor_repair_rule([pair])

    if rule is None:
        return None

    predicted = predict_anchor_repair_for_pair(
        rule,
        input_grid,
    )

    score = score_prediction_simple(
        predicted,
        output_grid,
    )

    return {
        "strategy": "anchor_repair_rule",
        "predicted": predicted,
        "prediction": predicted,
        "exact": predicted == output_grid,
        "score": score,
        "adjusted_score": score,
        "rule": rule,
        "mode": rule.get("rule_type"),
    }


def discover_anchor_repair_rule_for_task(train_pairs):
    return learn_anchor_repair_rule(train_pairs)


def apply_anchor_repair_rule(rule, input_grid):
    return predict_anchor_repair_for_pair(
        rule,
        input_grid,
    )