# reasoning/color_legend_recolor_rule.py

from core.scoring import score_prediction


# ============================================================
# BASIC GRID HELPERS
# ============================================================

def grid_shape(grid):
    if grid is None:
        return 0, 0

    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def bbox_for_cells(cells):
    if not cells:
        return None

    rows = [r for r, c in cells]
    cols = [c for r, c in cells]

    return {
        "top": min(rows),
        "left": min(cols),
        "bottom": max(rows),
        "right": max(cols),
        "height": max(rows) - min(rows) + 1,
        "width": max(cols) - min(cols) + 1,
    }


def crop_grid(grid, top, left, bottom, right):
    return [
        row[left:right + 1]
        for row in grid[top:bottom + 1]
    ]


def nonzero_colors(grid):
    colors = set()

    for row in grid:
        for value in row:
            if value != 0:
                colors.add(value)

    return colors


def color_counts(grid):
    counts = {}

    for row in grid:
        for value in row:
            if value == 0:
                continue

            counts[value] = counts.get(value, 0) + 1

    return counts


def most_common_nonzero_color(grid):
    counts = color_counts(grid)

    if not counts:
        return None

    return max(counts, key=counts.get)


def least_common_nonzero_color(grid):
    counts = color_counts(grid)

    if not counts:
        return None

    return min(counts, key=counts.get)


def only_nonzero_color(grid):
    colors = nonzero_colors(grid)

    if len(colors) != 1:
        return None

    return next(iter(colors))


# ============================================================
# ORDERED PANEL DETECTION
# ============================================================

def is_separator_column(grid, c):
    """
    Detect a likely vertical separator column.

    No hard-coded color.
    No hard-coded border.

    A separator column is:
        - mostly one non-zero color
        - tall relative to the grid
        - does not mix colors
    """

    h, w = grid_shape(grid)

    counts = {}

    for r in range(h):
        value = grid[r][c]

        if value == 0:
            continue

        counts[value] = counts.get(value, 0) + 1

    if not counts:
        return None

    dominant_color = max(counts, key=counts.get)
    dominant_count = counts[dominant_color]
    nonzero_count = sum(counts.values())

    if dominant_count != nonzero_count:
        return None

    if dominant_count < max(3, int(h * 0.75)):
        return None

    return dominant_color


def is_separator_row(grid, r):
    """
    Detect a likely horizontal separator row.

    No hard-coded color.
    No hard-coded border.
    """

    h, w = grid_shape(grid)

    counts = {}

    for c in range(w):
        value = grid[r][c]

        if value == 0:
            continue

        counts[value] = counts.get(value, 0) + 1

    if not counts:
        return None

    dominant_color = max(counts, key=counts.get)
    dominant_count = counts[dominant_color]
    nonzero_count = sum(counts.values())

    if dominant_count != nonzero_count:
        return None

    if dominant_count < max(3, int(w * 0.75)):
        return None

    return dominant_color


def build_panel_from_box(grid, top, left, bottom, right):
    h, w = grid_shape(grid)

    if h == 0 or w == 0:
        return None

    top = max(0, top)
    left = max(0, left)
    bottom = min(h - 1, bottom)
    right = min(w - 1, right)

    if bottom < top or right < left:
        return None

    panel_grid = crop_grid(grid, top, left, bottom, right)

    cells = []

    for rr, row in enumerate(panel_grid):
        for cc, value in enumerate(row):
            if value != 0:
                cells.append((top + rr, left + cc))

    if not cells:
        return None

    bbox = bbox_for_cells(cells)

    return {
        "grid": panel_grid,
        "cells": cells,
        "bbox": bbox,
        "top": top,
        "left": left,
        "bottom": bottom,
        "right": right,
        "height": bottom - top + 1,
        "width": right - left + 1,
        "area": len(cells),
    }


def split_panels_by_columns(grid):
    h, w = grid_shape(grid)

    separator_items = []

    for c in range(w):
        color = is_separator_column(grid, c)

        if color is not None:
            separator_items.append({
                "index": c,
                "color": color,
            })

    if not separator_items:
        return []

    by_color = {}

    for item in separator_items:
        by_color.setdefault(item["color"], []).append(item["index"])

    # Prefer the color that creates the most repeated separators.
    best_color = None
    best_cols = []

    for color, cols in by_color.items():
        cols = sorted(cols)

        if len(cols) > len(best_cols):
            best_color = color
            best_cols = cols

    if not best_cols:
        return []

    panels = []
    start_c = 0

    for sep_c in best_cols:
        panel = build_panel_from_box(
            grid=grid,
            top=0,
            left=start_c,
            bottom=h - 1,
            right=sep_c - 1,
        )

        if panel is not None:
            panels.append(panel)

        start_c = sep_c + 1

    panel = build_panel_from_box(
        grid=grid,
        top=0,
        left=start_c,
        bottom=h - 1,
        right=w - 1,
    )

    if panel is not None:
        panels.append(panel)

    return panels


def split_panels_by_rows(grid):
    h, w = grid_shape(grid)

    separator_items = []

    for r in range(h):
        color = is_separator_row(grid, r)

        if color is not None:
            separator_items.append({
                "index": r,
                "color": color,
            })

    if not separator_items:
        return []

    by_color = {}

    for item in separator_items:
        by_color.setdefault(item["color"], []).append(item["index"])

    best_color = None
    best_rows = []

    for color, rows in by_color.items():
        rows = sorted(rows)

        if len(rows) > len(best_rows):
            best_color = color
            best_rows = rows

    if not best_rows:
        return []

    panels = []
    start_r = 0

    for sep_r in best_rows:
        panel = build_panel_from_box(
            grid=grid,
            top=start_r,
            left=0,
            bottom=sep_r - 1,
            right=w - 1,
        )

        if panel is not None:
            panels.append(panel)

        start_r = sep_r + 1

    panel = build_panel_from_box(
        grid=grid,
        top=start_r,
        left=0,
        bottom=h - 1,
        right=w - 1,
    )

    if panel is not None:
        panels.append(panel)

    return panels


def find_divider_panels_auto(grid):
    """
    Find ordered panels without hard-coding color.

    It tries:
        horizontal layout: panels split by vertical separator columns
        vertical layout  : panels split by horizontal separator rows

    Returns the better panel list.
    """

    col_panels = split_panels_by_columns(grid)
    row_panels = split_panels_by_rows(grid)

    # Prefer whichever finds more ordered panels.
    if len(row_panels) > len(col_panels):
        return row_panels

    if len(col_panels) > len(row_panels):
        return col_panels

    # Tie-breaker: prefer 4 panels if one has it.
    if len(row_panels) == 4:
        return row_panels

    if len(col_panels) == 4:
        return col_panels

    return col_panels or row_panels


def find_ordered_panel_sets(grid):
    """
    Return both possible ordered panel interpretations.
    """

    sets = []

    col_panels = split_panels_by_columns(grid)

    if len(col_panels) >= 3:
        sets.append({
            "orientation": "horizontal",
            "panels": col_panels,
        })

    row_panels = split_panels_by_rows(grid)

    if len(row_panels) >= 3:
        sets.append({
            "orientation": "vertical",
            "panels": row_panels,
        })

    return sets


# ============================================================
# OLD COMPONENT DEBUG SUPPORT
# ============================================================

def find_nonzero_components(grid):
    """
    Kept for debug fallback.

    The actual rule prefers ordered panel splitting.
    """

    h, w = grid_shape(grid)
    seen = set()
    components = []

    for start_r in range(h):
        for start_c in range(w):
            if grid[start_r][start_c] == 0:
                continue

            if (start_r, start_c) in seen:
                continue

            stack = [(start_r, start_c)]
            seen.add((start_r, start_c))
            cells = []

            while stack:
                r, c = stack.pop()
                cells.append((r, c))

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = r + dr
                    nc = c + dc

                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue

                    if grid[nr][nc] == 0:
                        continue

                    if (nr, nc) in seen:
                        continue

                    seen.add((nr, nc))
                    stack.append((nr, nc))

            bbox = bbox_for_cells(cells)

            if bbox is None:
                continue

            components.append({
                "cells": cells,
                "bbox": bbox,
                "area": len(cells),
            })

    components.sort(
        key=lambda item: (
            item["bbox"]["top"],
            item["bbox"]["left"],
            item["bbox"]["height"] * item["bbox"]["width"],
        )
    )

    return components


def select_source_block(blocks, grid=None):
    """
    Kept for debug compatibility.

    Prefer largest multi-color component.
    """

    if not blocks:
        return None

    if grid is None:
        return max(blocks, key=lambda block: block.get("area", 0))

    multi = []

    for block in blocks:
        colors = set()

        for r, c in block["cells"]:
            value = grid[r][c]

            if value != 0:
                colors.add(value)

        if len(colors) >= 2:
            multi.append(block)

    if multi:
        return max(multi, key=lambda block: block.get("area", 0))

    return max(blocks, key=lambda block: block.get("area", 0))


# ============================================================
# MASK HELPERS
# ============================================================

def mask_from_panel_color(panel, color_key):
    grid = panel["grid"]

    color = get_panel_color_by_key(panel, color_key)

    if color is None:
        return None

    cells = []

    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == color:
                cells.append((r, c))

    if not cells:
        return None

    return cells_to_mask(cells)


def get_panel_color_by_key(panel, key):
    grid = panel["grid"]

    counts = color_counts(grid)

    if not counts:
        return None

    colors = sorted(counts)

    if key == "only":
        return only_nonzero_color(grid)

    if key == "least":
        return least_common_nonzero_color(grid)

    if key == "most":
        return most_common_nonzero_color(grid)

    if key.startswith("color_index_"):
        index = int(key.split("_")[-1])

        if index < 0 or index >= len(colors):
            return None

        return colors[index]

    return None


def cells_to_mask(cells):
    bbox = bbox_for_cells(cells)

    if bbox is None:
        return None

    out = [
        [0 for _ in range(bbox["width"])]
        for _ in range(bbox["height"])
    ]

    for r, c in cells:
        rr = r - bbox["top"]
        cc = c - bbox["left"]
        out[rr][cc] = 1

    return trim_empty_rows_cols(out)


def trim_empty_rows_cols(mask):
    cells = []

    for r, row in enumerate(mask):
        for c, value in enumerate(row):
            if value != 0:
                cells.append((r, c))

    if not cells:
        return None

    bbox = bbox_for_cells(cells)

    return [
        row[bbox["left"]:bbox["right"] + 1]
        for row in mask[bbox["top"]:bbox["bottom"] + 1]
    ]


def rotate_mask_90(mask):
    h = len(mask)
    w = len(mask[0]) if h else 0

    return [
        [
            mask[h - 1 - r][c]
            for r in range(h)
        ]
        for c in range(w)
    ]


def rotate_mask_180(mask):
    return rotate_mask_90(rotate_mask_90(mask))


def rotate_mask_270(mask):
    return rotate_mask_90(rotate_mask_180(mask))


def flip_mask_horizontal(mask):
    return [
        list(reversed(row))
        for row in mask
    ]


def transform_mask(mask, transform_name):
    if mask is None:
        return None

    if transform_name == "identity":
        return mask

    if transform_name == "rotate_90":
        return rotate_mask_90(mask)

    if transform_name == "rotate_180":
        return rotate_mask_180(mask)

    if transform_name == "rotate_270":
        return rotate_mask_270(mask)

    if transform_name == "flip_horizontal":
        return flip_mask_horizontal(mask)

    return None


def colorize_mask(mask, color):
    if mask is None:
        return None

    return [
        [
            color if value else 0
            for value in row
        ]
        for row in mask
    ]


def overlay_grid(base, piece, top, left):
    if base is None or piece is None:
        return None

    for r, row in enumerate(piece):
        for c, value in enumerate(row):
            if value == 0:
                continue

            rr = top + r
            cc = left + c

            if rr < 0 or rr >= len(base):
                continue

            if cc < 0 or cc >= len(base[0]):
                continue

            base[rr][cc] = value

    return base


def compose_two_pieces(piece_a, piece_b, axis, gap, align):
    if piece_a is None or piece_b is None:
        return None

    ha = len(piece_a)
    wa = len(piece_a[0]) if ha else 0
    hb = len(piece_b)
    wb = len(piece_b[0]) if hb else 0

    if ha == 0 or wa == 0 or hb == 0 or wb == 0:
        return None

    if axis == "horizontal":
        out_h = max(ha, hb)
        out_w = wa + gap + wb

        out = [
            [0 for _ in range(out_w)]
            for _ in range(out_h)
        ]

        if align == "start":
            top_a = 0
            top_b = 0
        elif align == "end":
            top_a = out_h - ha
            top_b = out_h - hb
        else:
            top_a = (out_h - ha) // 2
            top_b = (out_h - hb) // 2

        overlay_grid(out, piece_a, top_a, 0)
        overlay_grid(out, piece_b, top_b, wa + gap)

        return trim_empty_rows_cols(out)

    if axis == "vertical":
        out_h = ha + gap + hb
        out_w = max(wa, wb)

        out = [
            [0 for _ in range(out_w)]
            for _ in range(out_h)
        ]

        if align == "start":
            left_a = 0
            left_b = 0
        elif align == "end":
            left_a = out_w - wa
            left_b = out_w - wb
        else:
            left_a = (out_w - wa) // 2
            left_b = (out_w - wb) // 2

        overlay_grid(out, piece_a, 0, left_a)
        overlay_grid(out, piece_b, ha + gap, left_b)

        return trim_empty_rows_cols(out)

    return None


# ============================================================
# CANDIDATE APPLICATION
# ============================================================

PANEL_COLOR_KEYS = [
    "only",
    "least",
    "most",
    "color_index_0",
    "color_index_1",
    "color_index_2",
]

MASK_TRANSFORMS = [
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "flip_horizontal",
]

COMPOSE_AXES = [
    "horizontal",
    "vertical",
]

ALIGN_MODES = [
    "start",
    "center",
    "end",
]

GAPS = [
    0,
    1,
    2,
]


def resolve_panel_index(panels, index):
    if index >= 0:
        if index >= len(panels):
            return None

        return panels[index]

    real_index = len(panels) + index

    if real_index < 0 or real_index >= len(panels):
        return None

    return panels[real_index]


def apply_candidate_to_panel_set(panel_set, candidate):
    panels = panel_set["panels"]

    template_a_panel = resolve_panel_index(
        panels,
        candidate["template_a_panel_index"],
    )
    template_b_panel = resolve_panel_index(
        panels,
        candidate["template_b_panel_index"],
    )
    color_a_panel = resolve_panel_index(
        panels,
        candidate["color_a_panel_index"],
    )
    color_b_panel = resolve_panel_index(
        panels,
        candidate["color_b_panel_index"],
    )

    if template_a_panel is None:
        return None

    if template_b_panel is None:
        return None

    if color_a_panel is None:
        return None

    if color_b_panel is None:
        return None

    mask_a = mask_from_panel_color(
        template_a_panel,
        candidate["template_a_color_key"],
    )
    mask_b = mask_from_panel_color(
        template_b_panel,
        candidate["template_b_color_key"],
    )

    if mask_a is None or mask_b is None:
        return None

    mask_a = transform_mask(mask_a, candidate["template_a_transform"])
    mask_b = transform_mask(mask_b, candidate["template_b_transform"])

    if mask_a is None or mask_b is None:
        return None

    new_color_a = get_panel_color_by_key(
        color_a_panel,
        candidate["color_a_key"],
    )
    new_color_b = get_panel_color_by_key(
        color_b_panel,
        candidate["color_b_key"],
    )

    if new_color_a is None:
        return None

    if new_color_b is None:
        return None

    piece_a = colorize_mask(mask_a, new_color_a)
    piece_b = colorize_mask(mask_b, new_color_b)

    return compose_two_pieces(
        piece_a=piece_a,
        piece_b=piece_b,
        axis=candidate["compose_axis"],
        gap=candidate["gap"],
        align=candidate["align"],
    )


def apply_candidate_rule(input_grid, candidate):
    panel_sets = find_ordered_panel_sets(input_grid)

    for panel_set in panel_sets:
        if panel_set["orientation"] != candidate["panel_orientation"]:
            continue

        predicted = apply_candidate_to_panel_set(panel_set, candidate)

        if predicted is not None:
            return predicted

    return None


# ============================================================
# DISCOVERY
# ============================================================

def generate_candidate_rules():
    """
    Search a broad but still structured rule space.

    Assumption this family is allowed to learn:
        ordered panels contain:
            template A
            template B
            color A example
            color B example

    It does NOT hard-code:
        task id
        divider color
        horizontal vs vertical
        exact colors
        exact shapes
    """

    candidates = []

    panel_orientations = [
        "horizontal",
        "vertical",
    ]

    # Use first two panels as templates.
    template_panel_options = [
        (0, 1),
    ]

    # Use last two panels as output color examples.
    color_panel_options = [
        (-2, -1),
        (2, 3),
    ]

    for panel_orientation in panel_orientations:
        for template_a_panel_index, template_b_panel_index in template_panel_options:
            for color_a_panel_index, color_b_panel_index in color_panel_options:
                for template_a_color_key in PANEL_COLOR_KEYS:
                    for template_b_color_key in PANEL_COLOR_KEYS:
                        for color_a_key in PANEL_COLOR_KEYS:
                            for color_b_key in PANEL_COLOR_KEYS:
                                for template_a_transform in MASK_TRANSFORMS:
                                    for template_b_transform in MASK_TRANSFORMS:
                                        for compose_axis in COMPOSE_AXES:
                                            for gap in GAPS:
                                                for align in ALIGN_MODES:
                                                    candidates.append({
                                                        "family": "color_legend_recolor_rule",
                                                        "mode": "ordered_panel_template_recolor",
                                                        "panel_orientation": panel_orientation,
                                                        "template_a_panel_index": template_a_panel_index,
                                                        "template_b_panel_index": template_b_panel_index,
                                                        "color_a_panel_index": color_a_panel_index,
                                                        "color_b_panel_index": color_b_panel_index,
                                                        "template_a_color_key": template_a_color_key,
                                                        "template_b_color_key": template_b_color_key,
                                                        "color_a_key": color_a_key,
                                                        "color_b_key": color_b_key,
                                                        "template_a_transform": template_a_transform,
                                                        "template_b_transform": template_b_transform,
                                                        "compose_axis": compose_axis,
                                                        "gap": gap,
                                                        "align": align,
                                                    })

    return candidates


def discover_color_legend_recolor_rule_for_task(train_pairs, debug=False):
    if not train_pairs:
        return None

    candidates = generate_candidate_rules()

    best_rule = None
    best_exact_count = -1
    best_total_score = -10 ** 18
    best_results = None

    for candidate in candidates:
        exact_count = 0
        total_score = 0
        results = []

        for pair_index, pair in enumerate(train_pairs):
            predicted = apply_candidate_rule(
                input_grid=pair["input"],
                candidate=candidate,
            )

            expected = pair["output"]

            score = score_prediction(predicted, expected)
            exact = predicted == expected

            if exact:
                exact_count += 1

            total_score += score

            results.append({
                "pair_index": pair_index,
                "predicted": predicted,
                "score": score,
                "exact": exact,
            })

        if exact_count > best_exact_count:
            best_exact_count = exact_count
            best_total_score = total_score
            best_rule = candidate
            best_results = results

        elif exact_count == best_exact_count and total_score > best_total_score:
            best_total_score = total_score
            best_rule = candidate
            best_results = results

        if exact_count == len(train_pairs):
            break

    if best_rule is None:
        return None

    learned_rule = dict(best_rule)
    learned_rule["pair_count"] = len(train_pairs)
    learned_rule["exact_count"] = best_exact_count
    learned_rule["total_score"] = best_total_score
    learned_rule["results"] = best_results

    if debug:
        print()
        print("[COLOR LEGEND RECOLOR DEBUG]")
        print("-" * 60)
        print(f"Exact count : {best_exact_count} / {len(train_pairs)}")
        print(f"Total score : {best_total_score}")
        print(f"Rule        : {learned_rule}")

    # Trust only if it solves all visible training pairs.
    if best_exact_count != len(train_pairs):
        return None

    return learned_rule


def apply_color_legend_recolor_rule(task_rule, input_grid):
    if task_rule is None:
        return None

    return apply_candidate_rule(
        input_grid=input_grid,
        candidate=task_rule,
    )


def score_color_legend_recolor_rule_on_train(task_rule, train_pairs):
    if task_rule is None:
        return None

    total_score = 0
    exact_count = 0
    pair_count = 0
    results = []

    for pair_index, pair in enumerate(train_pairs):
        predicted = apply_color_legend_recolor_rule(
            task_rule=task_rule,
            input_grid=pair["input"],
        )

        expected = pair["output"]

        score = score_prediction(predicted, expected)
        exact = predicted == expected

        if exact:
            exact_count += 1

        pair_count += 1
        total_score += score

        results.append({
            "pair_index": pair_index,
            "predicted": predicted,
            "score": score,
            "exact": exact,
        })

    return {
        "strategy": "color_legend_recolor_rule",
        "task_rule": task_rule,
        "rule": task_rule,
        "pair_count": pair_count,
        "exact_count": exact_count,
        "total_raw_score": total_score,
        "total_adjusted_score": total_score,
        "results": results,
    }


def leave_one_out_color_legend_recolor_rule(train_pairs):
    total_score = 0
    exact_count = 0
    pair_count = 0
    results = []

    for hidden_index in range(len(train_pairs)):
        visible_pairs = [
            pair
            for idx, pair in enumerate(train_pairs)
            if idx != hidden_index
        ]

        hidden_pair = train_pairs[hidden_index]

        rule = discover_color_legend_recolor_rule_for_task(
            visible_pairs,
            debug=False,
        )

        predicted = apply_color_legend_recolor_rule(
            task_rule=rule,
            input_grid=hidden_pair["input"],
        )

        expected = hidden_pair["output"]

        score = score_prediction(predicted, expected)
        exact = predicted == expected

        if exact:
            exact_count += 1

        pair_count += 1
        total_score += score

        results.append({
            "pair_index": hidden_index,
            "predicted": predicted,
            "score": score,
            "exact": exact,
        })

    return {
        "strategy": "color_legend_recolor_rule",
        "pair_count": pair_count,
        "exact_count": exact_count,
        "total_raw_score": total_score,
        "total_adjusted_score": total_score,
        "results": results,
    }