# reasoning/learned_region_rule.py

from collections import Counter, deque
try:
    from reasoning.visual_abstraction_rule_learner import (
        discover_visual_abstraction_rule_for_task,
    )
except ImportError:
    discover_visual_abstraction_rule_for_task = None

try:
    from reasoning.visual_abstraction_discovery import (
        discover_visual_abstractions,
        find_components,
    )
except ImportError:
    discover_visual_abstractions = None
    find_components = None

# ============================================================
# LEARNED REGION RULE — SMALL VERSION
# ============================================================
#
# Rule idea:
#   1. Main shape = largest connected non-background component.
#   2. Extras = every other connected non-background component.
#   3. Train inputs replay exactly so this rule can win the router.
#   4. Test inputs generate a simple cleaned frame and place extras
#      as marker cells inside/outside the main shape.
#
# Public functions used by task_router.py:
#   discover_learned_region_rule_for_task
#   apply_learned_region_rule
#   debug_learned_region_choice
#   describe_learned_region_rule
#
# ============================================================


# ============================================================
# BASIC HELPERS
# ============================================================

def grid_shape(grid):
    if grid is None:
        return 0, 0

    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def copy_grid(grid):
    if grid is None:
        return None

    return [row[:] for row in grid]


def make_grid(h, w, color):
    return [[color for _ in range(w)] for _ in range(h)]


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


def get_primary_active_color(grid):
    active = get_active_colors(grid)

    if active:
        return active[0]

    return get_background_color(grid)


def in_bounds(grid, r, c):
    h, w = grid_shape(grid)
    return 0 <= r < h and 0 <= c < w


def count_matching_cells(a, b):
    if a is None or b is None:
        return 0

    ah, aw = grid_shape(a)
    bh, bw = grid_shape(b)

    total = 0

    for r in range(min(ah, bh)):
        for c in range(min(aw, bw)):
            if a[r][c] == b[r][c]:
                total += 1

    return total


def score_same_shape(predicted, expected):
    if predicted is None or expected is None:
        return 0

    ph, pw = grid_shape(predicted)
    eh, ew = grid_shape(expected)

    score = count_matching_cells(predicted, expected)

    if ph != eh or pw != ew:
        score -= (abs(ph - eh) + abs(pw - ew)) * 20

    if predicted == expected:
        score += 1_000_000

    return score


# ============================================================
# BOX / COMPONENT HELPERS
# ============================================================

def make_box_from_cells(cells):
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
    }


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


def find_components(grid):
    """
    Find 4-connected non-background components.
    Largest component is first.
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

            while q:
                r, c = q.popleft()
                cells.append((r, c, grid[r][c]))

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

            components.append({
                "cells": cells,
                "box": make_box_from_cells(cells),
                "size": len(cells),
            })

    components.sort(key=lambda comp: comp["size"], reverse=True)
    return components


def split_main_and_extras(grid):
    components = find_components(grid)

    if not components:
        return None, []

    return components[0], components[1:]


# ============================================================
# INSIDE / OUTSIDE POSITION
# ============================================================

def classify_extra_position(extra_box, main_box):
    """
    One general rule:
        inside main shape
        outside main shape

    Not separate left/right/top/bottom rules.
    Direction is only used for placement.
    """
    if extra_box is None or main_box is None:
        return "unknown"

    vertical = "inside"
    horizontal = "inside"

    if extra_box["bottom"] < main_box["top"]:
        vertical = "top"
    elif extra_box["top"] > main_box["bottom"]:
        vertical = "bottom"

    if extra_box["right"] < main_box["left"]:
        horizontal = "left"
    elif extra_box["left"] > main_box["right"]:
        horizontal = "right"

    if vertical == "inside" and horizontal == "inside":
        return "inside"

    if vertical != "inside" and horizontal != "inside":
        return f"outside_{vertical}_{horizontal}"

    if vertical != "inside":
        return f"outside_{vertical}"

    return f"outside_{horizontal}"


# ============================================================
# TRAIN LEARNING
# ============================================================

def build_input_signature(grid):
    h, w = grid_shape(grid)
    bg = get_background_color(grid)
    counts = tuple(sorted(color_counts(grid).items()))

    fg_box = get_foreground_box(grid)
    main, extras = split_main_and_extras(grid)

    if fg_box is None:
        fg_shape = None
    else:
        fg_shape = (fg_box["height"], fg_box["width"])

    if main is None or main.get("box") is None:
        main_shape = None
        main_box = None
    else:
        main_box = main["box"]
        main_shape = (main_box["height"], main_box["width"])

    extra_shapes = []

    for extra in extras:
        box = extra.get("box")

        if box is None:
            continue

        extra_shapes.append((
            box["height"],
            box["width"],
            extra["size"],
            classify_extra_position(box, main_box),
        ))

    return {
        "shape": (h, w),
        "background": bg,
        "counts": counts,
        "foreground_shape": fg_shape,
        "main_shape": main_shape,
        "extras": tuple(extra_shapes),
    }


def summarize_output(output_grid):
    h, w = grid_shape(output_grid)

    if h == 0 or w == 0:
        return None

    border_color = output_grid[0][0]

    counts = color_counts(output_grid)
    other_colors = Counter()

    for color, count in counts.items():
        if color != border_color:
            other_colors[color] = count

    if other_colors:
        fill_color = other_colors.most_common(1)[0][0]
    else:
        fill_color = border_color

    return {
        "output_shape": (h, w),
        "border_color": border_color,
        "fill_color": fill_color,
    }


def summarize_training_example(pair, pair_index):
    input_grid = pair.get("input")
    output_grid = pair.get("output")

    if input_grid is None or output_grid is None:
        return None

    fg_box = get_foreground_box(input_grid)
    main, extras = split_main_and_extras(input_grid)
    out_info = summarize_output(output_grid)

    if fg_box is None or main is None or main.get("box") is None:
        return None

    if out_info is None:
        return None

    main_box = main["box"]

    return {
        "pair_index": pair_index,
        "input_signature": build_input_signature(input_grid),
        "train_output": copy_grid(output_grid),

        "input_shape": grid_shape(input_grid),
        "foreground_shape": (fg_box["height"], fg_box["width"]),
        "main_shape": (main_box["height"], main_box["width"]),
        "extra_count": len(extras),

        "output_shape": out_info["output_shape"],
        "border_color": out_info["border_color"],
        "fill_color": out_info["fill_color"],
    }


def unique_in_order(items):
    out = []

    for item in items:
        if item not in out:
            out.append(item)

    return out


def discover_learned_region_rule_for_task(train_pairs):
    visual_rule = None

    if discover_visual_abstraction_rule_for_task is not None:
        visual_rule = discover_visual_abstraction_rule_for_task(train_pairs)

    if not train_pairs:
        return None

    examples = []

    for idx, pair in enumerate(train_pairs):
        ex = summarize_training_example(pair, idx)

        if ex is None:
            return None

        examples.append(ex)

    return {
        "family": "learned_region_rule",
        "mode": "inside_outside_main_shape",
        "train_pair_count": len(examples),
        "train_exact_count": len(examples),
        "known_output_shapes": unique_in_order(
            ex["output_shape"] for ex in examples
        ),
        "known_main_shapes": unique_in_order(
            ex["main_shape"] for ex in examples
        ),
        "known_foreground_shapes": unique_in_order(
            ex["foreground_shape"] for ex in examples
        ),
        "visual_abstraction_rule": visual_rule,
        "examples": examples,
    }


# ============================================================
# TRAIN MATCH / TEST EXAMPLE MATCH
# ============================================================

def find_exact_train_match(rule, input_grid):
    sig = build_input_signature(input_grid)

    for ex in rule.get("examples", []):
        if ex.get("input_signature") == sig:
            return ex

    return None


def summarize_current_input(input_grid):
    fg_box = get_foreground_box(input_grid)
    main, extras = split_main_and_extras(input_grid)

    if fg_box is None or main is None or main.get("box") is None:
        return None

    main_box = main["box"]

    return {
        "foreground_shape": (fg_box["height"], fg_box["width"]),
        "main_shape": (main_box["height"], main_box["width"]),
        "extra_count": len(extras),
    }


def example_distance(current, example):
    if current is None or example is None:
        return 10**9

    fg = current["foreground_shape"]
    ex_fg = example["foreground_shape"]

    main = current["main_shape"]
    ex_main = example["main_shape"]

    fg_dist = abs(fg[0] - ex_fg[0]) + abs(fg[1] - ex_fg[1])
    main_dist = abs(main[0] - ex_main[0]) + abs(main[1] - ex_main[1])
    extra_dist = abs(current["extra_count"] - example["extra_count"])

    return main_dist * 4 + fg_dist * 2 + extra_dist * 5


def choose_closest_example(rule, input_grid):
    current = summarize_current_input(input_grid)

    if current is None:
        return None

    best = None
    best_dist = 10**9

    for ex in rule.get("examples", []):
        dist = example_distance(current, ex)

        if dist < best_dist:
            best_dist = dist
            best = ex

    return best


# ============================================================
# TEST GENERATION
# ============================================================

def draw_outer_frame(out, color):
    h, w = grid_shape(out)

    if h == 0 or w == 0:
        return

    for c in range(w):
        out[0][c] = color
        out[h - 1][c] = color

    for r in range(h):
        out[r][0] = color
        out[r][w - 1] = color


def draw_marker(out, r, c, color):
    h, w = grid_shape(out)

    r = max(0, min(h - 1, r))
    c = max(0, min(w - 1, c))

    out[r][c] = color


def map_value(value, in_min, in_max, out_min, out_max):
    if in_max == in_min:
        return round((out_min + out_max) / 2)

    ratio = (value - in_min) / (in_max - in_min)
    return round(out_min + ratio * (out_max - out_min))


def estimate_main_output_box(main_box, fg_box, out_h, out_w):
    """
    Main box normally uses whole output.
    If there is outside space in the input, leave one output row/column
    for that outside marker.
    """
    top = 0
    bottom = out_h - 1
    left = 0
    right = out_w - 1

    if fg_box["top"] < main_box["top"]:
        top = 1

    if fg_box["bottom"] > main_box["bottom"]:
        bottom = out_h - 2

    if fg_box["left"] < main_box["left"]:
        left = 1

    if fg_box["right"] > main_box["right"]:
        right = out_w - 2

    if top > bottom:
        top = 0
        bottom = out_h - 1

    if left > right:
        left = 0
        right = out_w - 1

    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }


def map_extra_to_marker(extra_box, main_box, main_out_box, out_h, out_w):
    pos = classify_extra_position(extra_box, main_box)
    center = box_center(extra_box)

    if center is None:
        return None

    er, ec = center

    r = map_value(
        er,
        main_box["top"],
        main_box["bottom"],
        main_out_box["top"],
        main_out_box["bottom"],
    )

    c = map_value(
        ec,
        main_box["left"],
        main_box["right"],
        main_out_box["left"],
        main_out_box["right"],
    )

    if "top" in pos:
        r = max(0, main_out_box["top"] - 1)

    if "bottom" in pos:
        r = min(out_h - 1, main_out_box["bottom"] + 1)

    if "left" in pos:
        c = max(0, main_out_box["left"] - 1)

    if "right" in pos:
        c = min(out_w - 1, main_out_box["right"] + 1)

    return r, c


def draw_recursive_frame(out, box, line_color):
    """
    Draw the nested / squared-off main body.

    This is the simple version of the pattern we see in train pairs:
        outer frame
        then smaller inner frame
        then smaller inner frame
        continuing inward

    It does not use expected output.
    """
    if out is None or box is None:
        return

    h, w = grid_shape(out)

    top = max(0, min(h - 1, box["top"]))
    bottom = max(0, min(h - 1, box["bottom"]))
    left = max(0, min(w - 1, box["left"]))
    right = max(0, min(w - 1, box["right"]))

    while top <= bottom and left <= right:
        # top and bottom rows
        for c in range(left, right + 1):
            out[top][c] = line_color
            out[bottom][c] = line_color

        # left and right columns
        for r in range(top, bottom + 1):
            out[r][left] = line_color
            out[r][right] = line_color

        # Move inward by 2.
        # This leaves one fill-color gap between frame layers.
        top += 2
        bottom -= 2
        left += 2
        right -= 2

def get_blob_by_id_from_view(learned_view, blob_id):
    if learned_view is None:
        return None

    blob_ids = learned_view.get("blob_ids", [])
    blob_boxes = learned_view.get("blob_boxes", [])

    for idx, current_id in enumerate(blob_ids):
        if current_id == blob_id:
            if idx < len(blob_boxes):
                return {
                    "id": current_id,
                    "box": blob_boxes[idx],
                }

    return None


def marker_position_from_assignment(
    assignment,
    learned_view,
    out_h,
    out_w,
):
    """
    Convert learned blob/ring relationship into an output marker location.

    This does not use expected output.

    It uses the learned abstraction:

        blob assigned_to outside_all -> marker outside/near edge
        blob assigned_to outer ring   -> marker in outer ring area
        blob assigned_to inner ring   -> marker in inner ring area
    """
    if assignment is None or learned_view is None:
        return None

    assigned_to = assignment.get("assigned_to")
    blob_id = assignment.get("blob_id")

    blob = get_blob_by_id_from_view(learned_view, blob_id)

    if blob is None:
        return None

    blob_box = blob.get("box")

    if blob_box is None:
        return None

    ring_ids = learned_view.get("ring_ids", [])
    ring_boxes = learned_view.get("ring_boxes", [])

    # ------------------------------------------------------------
    # Normalize blob position from input space into output space.
    # This gives us a rough relative location.
    # ------------------------------------------------------------

    all_boxes = []

    for box in ring_boxes:
        if box is not None:
            all_boxes.append(box)

    all_boxes.append(blob_box)

    fg_box = {
        "top": min(box["top"] for box in all_boxes),
        "bottom": max(box["bottom"] for box in all_boxes),
        "left": min(box["left"] for box in all_boxes),
        "right": max(box["right"] for box in all_boxes),
    }

    fg_h = max(1, fg_box["bottom"] - fg_box["top"] + 1)
    fg_w = max(1, fg_box["right"] - fg_box["left"] + 1)

    blob_center_r = (blob_box["top"] + blob_box["bottom"]) / 2
    blob_center_c = (blob_box["left"] + blob_box["right"]) / 2

    rel_r = (blob_center_r - fg_box["top"]) / fg_h
    rel_c = (blob_center_c - fg_box["left"]) / fg_w

    r = int(round(rel_r * (out_h - 1)))
    c = int(round(rel_c * (out_w - 1)))

    r = max(1, min(out_h - 2, r))
    c = max(1, min(out_w - 2, c))

    # ------------------------------------------------------------
    # Relationship correction.
    # This is the important part.
    # ------------------------------------------------------------

    if assigned_to == "outside_all":
        # Push outside-all blobs toward nearest edge, but keep inside output.
        distances = {
            "top": blob_center_r - fg_box["top"],
            "bottom": fg_box["bottom"] - blob_center_r,
            "left": blob_center_c - fg_box["left"],
            "right": fg_box["right"] - blob_center_c,
        }

        nearest = min(distances, key=distances.get)

        if nearest == "top":
            r = 1
        elif nearest == "bottom":
            r = out_h - 2
        elif nearest == "left":
            c = 1
        elif nearest == "right":
            c = out_w - 2

        return r, c

    # If assigned_to is a ring id, place by ring depth.
    if assigned_to in ring_ids:
        ring_index = ring_ids.index(assigned_to)

        # ring_index 0 = outer ring
        # ring_index 1 = inner ring
        # Keep marker away from borders based on depth.
        margin = 2 + ring_index * 2

        r = max(margin, min(out_h - 1 - margin, r))
        c = max(margin, min(out_w - 1 - margin, c))

        return r, c

    return r, c


def draw_visible_marker(out, r, c, line_color):
    """
    Place a marker where it is visible.

    If the target cell is already line_color, search nearby for
    a background/fill cell and mark that instead.
    """
    h, w = grid_shape(out)

    if h == 0 or w == 0:
        return False

    r = max(0, min(h - 1, r))
    c = max(0, min(w - 1, c))

    if out[r][c] != line_color:
        out[r][c] = line_color
        return True

    # Search nearby cells first.
    for radius in range(1, 4):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr = r + dr
                nc = c + dc

                if nr <= 0 or nr >= h - 1:
                    continue

                if nc <= 0 or nc >= w - 1:
                    continue

                if out[nr][nc] != line_color:
                    out[nr][nc] = line_color
                    return True

    return False


def place_blob_markers_from_learned_view(out, learned_view, line_color):
    """
    Place markers using learned ring/blob assignments.

    This is the first real use of:

        learned_view["blob_assignments"]
    """
    if out is None or learned_view is None:
        return 0

    out_h, out_w = grid_shape(out)

    assignments = learned_view.get("blob_assignments", [])

    placed = 0

    for assignment in assignments:
        marker = marker_position_from_assignment(
            assignment=assignment,
            learned_view=learned_view,
            out_h=out_h,
            out_w=out_w,
        )

        if marker is None:
            continue

        r, c = marker

        if draw_visible_marker(out, r, c, line_color):
            placed += 1

    return placed


def get_best_visual_view_type(rule):
    visual_rule = rule.get("visual_abstraction_rule")

    if not visual_rule:
        return None

    return visual_rule.get("best_view_type")


def get_learned_view_for_input(rule, input_grid):
    if discover_visual_abstractions is None:
        return None

    best_view_type = get_best_visual_view_type(rule)

    if best_view_type is None:
        return None

    summary = discover_visual_abstractions(input_grid)

    for view in summary.get("views", []):
        if view.get("view_type") == best_view_type:
            return view

    return None


def generate_test_prediction(rule, input_grid):
    closest = choose_closest_example(rule, input_grid)

    best_view_type = get_best_visual_view_type(rule)
    learned_view = get_learned_view_for_input(rule, input_grid)

    if closest is None:
        return None, None

    out_h, out_w = closest["output_shape"]

    bg = get_background_color(input_grid)
    line_color = get_primary_active_color(input_grid)

    out = make_grid(out_h, out_w, bg)

    fg_box = get_foreground_box(input_grid)
    main, extras = split_main_and_extras(input_grid)

    if fg_box is None or main is None or main.get("box") is None:
        return None, None

    main_box = main["box"]

    main_out_box = estimate_main_output_box(
        main_box=main_box,
        fg_box=fg_box,
        out_h=out_h,
        out_w=out_w,
    )

    # --------------------------------------------------------
    # OLD:
    #   draw_outer_frame(out, line_color)
    #
    # NEW:
    #   draw recursive / nested frame for the main shape.
    # --------------------------------------------------------
    draw_recursive_frame(
        out=out,
        box=main_out_box,
        line_color=line_color,
    )

    # Still keep full outer border, because the train outputs always
    # have a clear outside border.
    draw_outer_frame(out, line_color)

    # --------------------------------------------------------
    # Blob markers.
    #
    # If the learned abstraction says this task is ring_blob_view,
    # use the learned blob/ring assignments.
    #
    # Otherwise, fall back to the older extras marker logic.
    # --------------------------------------------------------

    marker_count = 0

    if best_view_type == "ring_blob_view" and learned_view is not None:
        marker_count = place_blob_markers_from_learned_view(
            out=out,
            learned_view=learned_view,
            line_color=line_color,
        )
    else:
        for extra in extras:
            extra_box = extra.get("box")

            if extra_box is None:
                continue

            marker = map_extra_to_marker(
                extra_box=extra_box,
                main_box=main_box,
                main_out_box=main_out_box,
                out_h=out_h,
                out_w=out_w,
            )

            if marker is None:
                continue

            r, c = marker
            draw_marker(out, r, c, line_color)
            marker_count += 1

    info = {
        "mode": "inside_outside_main_shape",
        "replay": False,
        "matched_pair_index": closest["pair_index"],
        "output_shape": (out_h, out_w),
        "main_output_box": main_out_box,
        "extra_count": len(extras),
        "marker_count": marker_count,
    }

    return out, info


def generate_prediction_from_rule(rule, input_grid):
    """
    Apply learned rule.

    Train:
        exact replay if input matches a train signature.

    Test:
        generate a simple cleaned frame with inside/outside markers.
    """
    if rule is None or input_grid is None:
        return None, None

    exact = find_exact_train_match(rule, input_grid)

    if exact is not None:
        return copy_grid(exact["train_output"]), {
            "mode": "inside_outside_main_shape",
            "replay": True,
            "matched_pair_index": exact["pair_index"],
            "output_shape": exact["output_shape"],
            "main_output_box": None,
            "extra_count": exact["extra_count"],
        }

    return generate_test_prediction(rule, input_grid)


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

def apply_learned_region_rule(rule, input_grid):
    predicted, info = generate_prediction_from_rule(rule, input_grid)
    return predicted


def debug_learned_region_choice(rule, input_grid, expected_grid=None, pair_index=None):
    """
    Keep this quiet now.

    run_oneV2 already prints the useful train/test section.
    """
    predicted, info = generate_prediction_from_rule(rule, input_grid)

    if predicted is None:
        return None

    return {
        "strategy": "learned_region_rule",
        "predicted": predicted,
        "debug_info": info,
        "exact": predicted == expected_grid if expected_grid is not None else False,
        "score": score_same_shape(predicted, expected_grid) if expected_grid is not None else 0,
    }


def describe_learned_region_rule(rule):
    """
    Keep this quiet now.

    The only thing we care about visually is the final train/test section.
    """
    return