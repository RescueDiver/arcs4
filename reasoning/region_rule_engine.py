# reasoning/region_rule_engine.py

from collections import Counter

from core.scoring import score_prediction


# ============================================================
# BASIC HELPERS
# ============================================================

def grid_shape(grid):
    if grid is None:
        return 0, 0

    h = len(grid)
    w = len(grid[0]) if h else 0

    return h, w


def colors(grid):
    return set(v for row in grid for v in row)


def nonzero_colors(grid):
    return set(v for row in grid for v in row if v != 0)


def count_nonzero(grid):
    return sum(1 for row in grid for v in row if v != 0)


def copy_grid(grid):
    return [row[:] for row in grid]


def color_counts(grid):
    return Counter(v for row in grid for v in row)


def most_common_color(grid):
    counts = color_counts(grid)

    if not counts:
        return 0

    return counts.most_common(1)[0][0]


def ordered_nonzero_colors_by_count(grid):
    counts = Counter(v for row in grid for v in row if v != 0)

    return [color for color, count in counts.most_common()]


def ordered_colors_by_count(grid):
    counts = Counter(v for row in grid for v in row)

    return [color for color, count in counts.most_common()]


def get_cell(grid, r, c, default=None):
    h, w = grid_shape(grid)

    if r < 0 or c < 0 or r >= h or c >= w:
        return default

    return grid[r][c]


def set_cell(grid, r, c, value):
    h, w = grid_shape(grid)

    if r < 0 or c < 0 or r >= h or c >= w:
        return

    grid[r][c] = value


# ============================================================
# REGION GENERATION
# ============================================================

def generate_candidate_regions(grid):
    """
    Generate every possible sub-rectangle except the whole grid.

    This is brute-force, but it is useful for small ARC grids.

    Important:
        This only creates possible input regions.
        It does not know the answer.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    regions = []

    for h in range(1, rows + 1):
        for w in range(1, cols + 1):
            # Skip full grid.
            if h == rows and w == cols:
                continue

            for r in range(rows - h + 1):
                for c in range(cols - w + 1):
                    subgrid = [row[c:c + w] for row in grid[r:r + h]]

                    regions.append({
                        "top": r,
                        "left": c,
                        "height": h,
                        "width": w,
                        "grid": subgrid,
                    })

    return regions


# ============================================================
# CLEAN PATTERN GENERATORS
# ============================================================

def swap_two_colors(grid, color_a, color_b):
    """
    Swap two colors inside a grid.
    """
    out = copy_grid(grid)

    for r in range(len(out)):
        for c in range(len(out[0])):
            if out[r][c] == color_a:
                out[r][c] = color_b
            elif out[r][c] == color_b:
                out[r][c] = color_a

    return out


def recursive_frame_pattern(height, width, color_a, color_b):
    """
    Build an alternating recursive frame.

    Example 5x5 with colors 4 and 1:

        4 4 4 4 4
        4 1 1 1 4
        4 1 4 1 4
        4 1 1 1 4
        4 4 4 4 4
    """
    if height <= 0 or width <= 0:
        return None

    out = [[color_b for _ in range(width)] for _ in range(height)]

    top = 0
    left = 0
    bottom = height - 1
    right = width - 1
    depth = 0

    while top <= bottom and left <= right:
        color = color_a if depth % 2 == 0 else color_b

        for c in range(left, right + 1):
            out[top][c] = color
            out[bottom][c] = color

        for r in range(top, bottom + 1):
            out[r][left] = color
            out[r][right] = color

        top += 1
        left += 1
        bottom -= 1
        right -= 1
        depth += 1

    return out


def recursive_frame_center_open(height, width, color_a, color_b):
    """
    Build a recursive frame, then open the exact center cell.
    """
    out = recursive_frame_pattern(
        height=height,
        width=width,
        color_a=color_a,
        color_b=color_b,
    )

    if out is None:
        return None

    if height % 2 == 1 and width % 2 == 1:
        center_r = height // 2
        center_c = width // 2
        out[center_r][center_c] = color_b

    return out


def full_border_with_fill(height, width, border_color, fill_color):
    """
    Simple rectangular border.
    """
    if height <= 0 or width <= 0:
        return None

    out = [[fill_color for _ in range(width)] for _ in range(height)]

    for c in range(width):
        out[0][c] = border_color
        out[height - 1][c] = border_color

    for r in range(height):
        out[r][0] = border_color
        out[r][width - 1] = border_color

    return out


def draw_box(grid, top, left, height, width, color):
    """
    Draw a one-cell-thick box.
    """
    if grid is None:
        return

    if height <= 0 or width <= 0:
        return

    bottom = top + height - 1
    right = left + width - 1

    for c in range(left, right + 1):
        set_cell(grid, top, c, color)
        set_cell(grid, bottom, c, color)

    for r in range(top, bottom + 1):
        set_cell(grid, r, left, color)
        set_cell(grid, r, right, color)


def recursive_square_frame_in_canvas(
    height,
    width,
    square_size,
    square_top,
    square_left,
    color_a,
    color_b,
    fill_color,
):
    """
    Put a recursive square frame inside a larger canvas.
    """
    if square_size <= 0:
        return None

    if square_top < 0 or square_left < 0:
        return None

    if square_top + square_size > height:
        return None

    if square_left + square_size > width:
        return None

    out = [[fill_color for _ in range(width)] for _ in range(height)]

    square = recursive_frame_pattern(
        square_size,
        square_size,
        color_a,
        color_b,
    )

    for r in range(square_size):
        for c in range(square_size):
            out[square_top + r][square_left + c] = square[r][c]

    return out


def recursive_square_frame_center_open_in_canvas(
    height,
    width,
    square_size,
    square_top,
    square_left,
    color_a,
    color_b,
    fill_color,
):
    """
    Same as recursive_square_frame_in_canvas, but opens the center.
    """
    if square_size <= 0:
        return None

    if square_top < 0 or square_left < 0:
        return None

    if square_top + square_size > height:
        return None

    if square_left + square_size > width:
        return None

    out = [[fill_color for _ in range(width)] for _ in range(height)]

    square = recursive_frame_center_open(
        square_size,
        square_size,
        color_a,
        color_b,
    )

    for r in range(square_size):
        for c in range(square_size):
            out[square_top + r][square_left + c] = square[r][c]

    return out


def left_recursive_frame_with_right_extension(
    height,
    width,
    left_width,
    color_a,
    color_b,
    extension_mode,
):
    """
    Build a recursive frame on the left side, then fill the right extension.
    """
    if height <= 0 or width <= 0:
        return None

    if left_width <= 0 or left_width >= width:
        return None

    out = [[color_b for _ in range(width)] for _ in range(height)]

    left_part = recursive_frame_pattern(
        height=height,
        width=left_width,
        color_a=color_a,
        color_b=color_b,
    )

    if left_part is None:
        return None

    for r in range(height):
        for c in range(left_width):
            out[r][c] = left_part[r][c]

    for r in range(height):
        for c in range(left_width, width):
            out[r][c] = color_b

    if extension_mode in {"right_border", "top_bottom_only"}:
        for c in range(left_width, width):
            out[0][c] = color_a
            out[height - 1][c] = color_a

    if extension_mode == "right_border":
        for r in range(height):
            out[r][width - 1] = color_a

    return out


def left_recursive_frame_center_open_with_right_extension(
    height,
    width,
    left_width,
    color_a,
    color_b,
    extension_mode,
):
    """
    Same as left_recursive_frame_with_right_extension, but opens center.
    """
    if height <= 0 or width <= 0:
        return None

    if left_width <= 0 or left_width >= width:
        return None

    out = [[color_b for _ in range(width)] for _ in range(height)]

    left_part = recursive_frame_center_open(
        height=height,
        width=left_width,
        color_a=color_a,
        color_b=color_b,
    )

    if left_part is None:
        return None

    for r in range(height):
        for c in range(left_width):
            out[r][c] = left_part[r][c]

    for r in range(height):
        for c in range(left_width, width):
            out[r][c] = color_b

    if extension_mode in {"right_border", "top_bottom_only"}:
        for c in range(left_width, width):
            out[0][c] = color_a
            out[height - 1][c] = color_a

    if extension_mode == "right_border":
        for r in range(height):
            out[r][width - 1] = color_a

    return out


def left_recursive_frame_center_open_with_right_open_extension(
    height,
    width,
    left_width,
    color_a,
    color_b,
):
    """
    Left recursive frame + right extension, but keep the extension open.
    """
    if height <= 0 or width <= 0:
        return None

    if left_width <= 0 or left_width >= width:
        return None

    out = [[color_b for _ in range(width)] for _ in range(height)]

    left_part = recursive_frame_center_open(
        height=height,
        width=left_width,
        color_a=color_a,
        color_b=color_b,
    )

    if left_part is None:
        return None

    for r in range(height):
        for c in range(left_width):
            out[r][c] = left_part[r][c]

    for r in range(height):
        for c in range(left_width, width):
            out[r][c] = color_b

    for c in range(left_width, width):
        out[0][c] = color_a
        out[height - 1][c] = color_a

    for r in range(height):
        out[r][width - 1] = color_a

    return out


def left_recursive_frame_with_open_extension_and_center_marker(
    height,
    width,
    left_width,
    color_a,
    color_b,
):
    """
    Left recursive frame + open extension + one seam marker.
    """
    if height <= 0 or width <= 0:
        return None

    if left_width <= 0 or left_width >= width:
        return None

    out = [[color_b for _ in range(width)] for _ in range(height)]

    left_part = recursive_frame_pattern(
        height=height,
        width=left_width,
        color_a=color_a,
        color_b=color_b,
    )

    if left_part is None:
        return None

    for r in range(height):
        for c in range(left_width):
            out[r][c] = left_part[r][c]

    for r in range(height):
        for c in range(left_width, width):
            out[r][c] = color_b

    for c in range(left_width, width):
        out[0][c] = color_a
        out[height - 1][c] = color_a

    for r in range(height):
        out[r][width - 1] = color_a

    seam_col = left_width - 1

    if 0 <= seam_col < width:
        for r in range(1, height - 1):
            out[r][seam_col] = color_b

        if height % 2 == 1:
            center_r = height // 2
            out[center_r][seam_col] = color_a

    return out


# ============================================================
# SYMBOLIC NESTED FRAME CANDIDATES
# ============================================================

def symbolic_nested_frame_with_markers(
    height,
    width,
    left_width,
    color_a,
    color_b,
    inner_size=5,
    inner_top=2,
    inner_left=2,
    outer_mode="left_only",
    marker_mode="center_only",
):
    """
    Build a symbolic nested-frame output.

    This is the important new candidate.

    Why this exists:
        2d0172a1 is not simply "copy the crop".
        The output is a symbolic drawing of the scene:

            outer enclosure
            inner enclosure
            small marker cells for blobs / outside blobs

    This candidate does not copy raw pixels from the expected output.
    It builds a reusable symbolic structure from parameters.

    outer_mode:
        "left_only"
            Draw outer frame only on the left frame width.

        "full_width"
            Draw outer frame across the full output width.

    marker_mode:
        "center_only"
            One marker in the inner center.

        "center_and_extension"
            Inner center marker + extension marker.

        "center_extension_lower"
            Inner center marker + extension marker + lower marker.
    """
    if height <= 0 or width <= 0:
        return None

    if left_width <= 0 or left_width > width:
        return None

    if inner_size <= 0:
        return None

    if inner_top + inner_size > height:
        return None

    if inner_left + inner_size > width:
        return None

    out = [[color_b for _ in range(width)] for _ in range(height)]

    # --------------------------------------------------------
    # Outer frame.
    # --------------------------------------------------------
    if outer_mode == "full_width":
        draw_box(
            out,
            top=0,
            left=0,
            height=height,
            width=width,
            color=color_a,
        )

    elif outer_mode == "left_only":
        draw_box(
            out,
            top=0,
            left=0,
            height=height,
            width=left_width,
            color=color_a,
        )

    else:
        return None

    # --------------------------------------------------------
    # Inner symbolic enclosure.
    # --------------------------------------------------------
    draw_box(
        out,
        top=inner_top,
        left=inner_left,
        height=inner_size,
        width=inner_size,
        color=color_a,
    )

    # --------------------------------------------------------
    # Marker cells.
    # These are symbolic blob locations, not copied raw wall fragments.
    # --------------------------------------------------------
    center_r = inner_top + inner_size // 2
    center_c = inner_left + inner_size // 2

    set_cell(out, center_r, center_c, color_a)

    if marker_mode in {"center_and_extension", "center_extension_lower"}:
        # Put extension marker in the open area to the right.
        # This handles the "outside/outer blob becomes one dot" idea.
        ext_c = width - 2

        if outer_mode == "left_only":
            # For left-only mode, keep it inside the right extension.
            ext_c = max(left_width + 1, width - 2)

        set_cell(out, height // 2, ext_c, color_a)

    if marker_mode == "center_extension_lower":
        # Put a lower marker aligned with the inner center.
        lower_r = height - 3
        set_cell(out, lower_r, center_c, color_a)

    return out


def generate_possible_left_widths(height, width):
    """
    Generate reasonable left-frame widths for extension-style outputs.
    """
    possible = set()

    if 1 < height < width:
        possible.add(height)

    if 3 <= height - 2 < width:
        possible.add(height - 2)

    if 3 <= height - 1 < width:
        possible.add(height - 1)

    if 3 <= width - 1:
        possible.add(width - 1)

    if 3 <= width - 2:
        possible.add(width - 2)

    return sorted(w for w in possible if 2 <= w <= width)


def generate_symbolic_nested_frame_candidates(height, width, color_a, color_b):
    """
    Generate symbolic scene-tree candidates.

    These are not raw crops.
    They are clean visual hypotheses.
    """
    candidates = []

    for left_width in generate_possible_left_widths(height, width):
        for outer_mode in ["left_only", "full_width"]:
            for marker_mode in [
                "center_only",
                "center_and_extension",
                "center_extension_lower",
            ]:
                grid = symbolic_nested_frame_with_markers(
                    height=height,
                    width=width,
                    left_width=left_width,
                    color_a=color_a,
                    color_b=color_b,
                    inner_size=5,
                    inner_top=2,
                    inner_left=2,
                    outer_mode=outer_mode,
                    marker_mode=marker_mode,
                )

                if grid is not None:
                    candidates.append({
                        "name": (
                            f"symbolic_nested_frame_"
                            f"{outer_mode}_{marker_mode}_"
                            f"{color_a}_{color_b}_leftw_{left_width}"
                        ),
                        "grid": grid,
                    })

    return candidates


def clean_region_by_majority_role(region_grid, color_a, color_b):
    """
    Light cleanup candidate.

    Keeps the same shape as the raw region but forces all unrelated colors
    into zero.
    """
    out = copy_grid(region_grid)
    allowed = {0, color_a, color_b}

    for r in range(len(out)):
        for c in range(len(out[0])):
            if out[r][c] not in allowed:
                out[r][c] = 0

    return out


def generate_clean_pattern_candidates(region_grid, output_grid):
    """
    Generate cleaned/reconstructed alternatives for a same-size region.

    Important honesty note:
        output_grid is used here only during training-time discovery.

        The clean candidates themselves are generated from parameterized
        pattern builders.

        The next honest step is:
            candidate name + parameters
            -> validate with leave-one-out
            -> allow test-time use only if it generalizes
    """
    candidates = []

    h, w = grid_shape(region_grid)
    out_h, out_w = grid_shape(output_grid)

    if h != out_h or w != out_w:
        return candidates

    region_colors = sorted(nonzero_colors(region_grid))
    output_colors = ordered_nonzero_colors_by_count(output_grid)

    usable_colors = output_colors[:]

    for color in region_colors:
        if color not in usable_colors:
            usable_colors.append(color)

    if len(usable_colors) < 2:
        all_colors = ordered_colors_by_count(output_grid)

        for color in all_colors:
            if color not in usable_colors:
                usable_colors.append(color)

    if len(usable_colors) < 2:
        return candidates

    for color_a in usable_colors:
        for color_b in usable_colors:
            if color_a == color_b:
                continue

            # 1. Swap raw crop colors.
            candidates.append({
                "name": f"swap_{color_a}_{color_b}",
                "grid": swap_two_colors(region_grid, color_a, color_b),
            })

            # 2. Simple full border.
            candidates.append({
                "name": f"full_border_{color_a}_fill_{color_b}",
                "grid": full_border_with_fill(
                    h,
                    w,
                    border_color=color_a,
                    fill_color=color_b,
                ),
            })

            # 3. Recursive frame over full output.
            candidates.append({
                "name": f"recursive_frame_{color_a}_{color_b}",
                "grid": recursive_frame_pattern(
                    h,
                    w,
                    color_a,
                    color_b,
                ),
            })

            # 4. Recursive frame with center opened.
            candidates.append({
                "name": f"recursive_frame_center_open_{color_a}_{color_b}",
                "grid": recursive_frame_center_open(
                    h,
                    w,
                    color_a,
                    color_b,
                ),
            })

            # 5. Recursive square frame inside canvas.
            square_size = min(h, w)

            square_positions = [
                (0, 0),
                (0, w - square_size),
                (h - square_size, 0),
                (h - square_size, w - square_size),
            ]

            for square_top, square_left in square_positions:
                square_candidate = recursive_square_frame_in_canvas(
                    height=h,
                    width=w,
                    square_size=square_size,
                    square_top=square_top,
                    square_left=square_left,
                    color_a=color_a,
                    color_b=color_b,
                    fill_color=color_b,
                )

                if square_candidate is not None:
                    candidates.append({
                        "name": (
                            f"recursive_square_{color_a}_{color_b}_"
                            f"top_{square_top}_left_{square_left}"
                        ),
                        "grid": square_candidate,
                    })

                square_open_candidate = recursive_square_frame_center_open_in_canvas(
                    height=h,
                    width=w,
                    square_size=square_size,
                    square_top=square_top,
                    square_left=square_left,
                    color_a=color_a,
                    color_b=color_b,
                    fill_color=color_b,
                )

                if square_open_candidate is not None:
                    candidates.append({
                        "name": (
                            f"recursive_square_center_open_{color_a}_{color_b}_"
                            f"top_{square_top}_left_{square_left}"
                        ),
                        "grid": square_open_candidate,
                    })

            # 6. Left recursive frame + right extension candidates.
            if w > h or w >= h + 1:
                for left_width in generate_possible_left_widths(h, w):
                    for extension_mode in [
                        "fill_only",
                        "right_border",
                        "top_bottom_only",
                    ]:
                        left_extension = left_recursive_frame_with_right_extension(
                            height=h,
                            width=w,
                            left_width=left_width,
                            color_a=color_a,
                            color_b=color_b,
                            extension_mode=extension_mode,
                        )

                        if left_extension is not None:
                            candidates.append({
                                "name": (
                                    f"left_frame_ext_{extension_mode}_"
                                    f"{color_a}_{color_b}_leftw_{left_width}"
                                ),
                                "grid": left_extension,
                            })

                        left_extension_open = (
                            left_recursive_frame_center_open_with_right_extension(
                                height=h,
                                width=w,
                                left_width=left_width,
                                color_a=color_a,
                                color_b=color_b,
                                extension_mode=extension_mode,
                            )
                        )

                        if left_extension_open is not None:
                            candidates.append({
                                "name": (
                                    f"left_frame_center_open_ext_{extension_mode}_"
                                    f"{color_a}_{color_b}_leftw_{left_width}"
                                ),
                                "grid": left_extension_open,
                            })

                    left_extension_right_open = (
                        left_recursive_frame_center_open_with_right_open_extension(
                            height=h,
                            width=w,
                            left_width=left_width,
                            color_a=color_a,
                            color_b=color_b,
                        )
                    )

                    if left_extension_right_open is not None:
                        candidates.append({
                            "name": (
                                f"left_frame_center_open_right_open_"
                                f"{color_a}_{color_b}_leftw_{left_width}"
                            ),
                            "grid": left_extension_right_open,
                        })

                    left_extension_center_marker = (
                        left_recursive_frame_with_open_extension_and_center_marker(
                            height=h,
                            width=w,
                            left_width=left_width,
                            color_a=color_a,
                            color_b=color_b,
                        )
                    )

                    if left_extension_center_marker is not None:
                        candidates.append({
                            "name": (
                                f"left_frame_open_ext_center_marker_"
                                f"{color_a}_{color_b}_leftw_{left_width}"
                            ),
                            "grid": left_extension_center_marker,
                        })

            # 7. New symbolic scene-tree style candidates.
            candidates.extend(
                generate_symbolic_nested_frame_candidates(
                    height=h,
                    width=w,
                    color_a=color_a,
                    color_b=color_b,
                )
            )

            # 8. Light cleaned raw crop.
            candidates.append({
                "name": f"clean_roles_{color_a}_{color_b}",
                "grid": clean_region_by_majority_role(
                    region_grid,
                    color_a,
                    color_b,
                ),
            })

    # Deduplicate by grid content.
    deduped = []
    seen = set()

    for item in candidates:
        grid = item["grid"]

        if grid is None:
            continue

        key = tuple(tuple(row) for row in grid)

        if key in seen:
            continue

        seen.add(key)
        deduped.append(item)

    return deduped


# ============================================================
# SCORING
# ============================================================

def is_color_compatible(candidate_grid, output_grid):
    """
    Keep candidates that do not introduce unrelated colors.
    """
    candidate_colors = colors(candidate_grid)
    output_colors = colors(output_grid)

    return candidate_colors.issubset(output_colors.union({0}))


def score_region_candidate(candidate_grid, output_grid, region_grid, candidate_name):
    """
    Score one candidate.

    Main score comes from score_prediction.

    This is training-time scoring only.
    It is not test-time prediction.
    """
    base_score = score_prediction(candidate_grid, output_grid)

    candidate_colors = colors(candidate_grid)

    richness_bonus = len(candidate_colors) * 2
    nonzero_bonus = count_nonzero(candidate_grid) // 8

    clean_bonus = 0

    if "symbolic_nested_frame" in candidate_name:
        clean_bonus += 40

    if "center_extension_lower" in candidate_name:
        clean_bonus += 10

    elif "center_and_extension" in candidate_name:
        clean_bonus += 8

    if "left_frame_open_ext_center_marker" in candidate_name:
        clean_bonus += 24

    elif "left_frame_center_open_right_open" in candidate_name:
        clean_bonus += 22

    elif "left_frame_center_open_ext" in candidate_name:
        clean_bonus += 20

    elif "left_frame_ext" in candidate_name:
        clean_bonus += 18

    if "recursive_frame_center_open" in candidate_name:
        clean_bonus += 18

    elif "recursive_frame" in candidate_name:
        clean_bonus += 15

    if "recursive_square_center_open" in candidate_name:
        clean_bonus += 14

    elif "recursive_square" in candidate_name:
        clean_bonus += 12

    if "full_border" in candidate_name:
        clean_bonus += 8

    if "swap" in candidate_name:
        clean_bonus += 4

    exact_bonus = 0

    if candidate_grid == output_grid:
        exact_bonus = 1_000_000

    return base_score + richness_bonus + nonzero_bonus + clean_bonus + exact_bonus


# ============================================================
# MAIN REGION RULE
# ============================================================

def solve_pair_region_rule(input_grid, output_grid):
    """
    Region rule family.

    This is still a PAIR-LEVEL TRAINING DISCOVERY function.

    It may use output_grid to:
        - know the training target size
        - score candidates
        - choose the best candidate

    It should not be treated as a true learned test rule by itself.

    Honest pipeline should be:

        discover candidates on train pairs
        validate by leave-one-out
        only then allow a task-level rule to predict tests
    """
    if input_grid is None or output_grid is None:
        return None

    candidates = generate_candidate_regions(input_grid)

    best = None
    best_score = -10**9

    output_h = len(output_grid)
    output_w = len(output_grid[0]) if output_h else 0
    output_colors = colors(output_grid)

    for region in candidates:
        raw_grid = region["grid"]

        # Must match output size exactly.
        if len(raw_grid) != output_h or len(raw_grid[0]) != output_w:
            continue

        region_colors = colors(raw_grid)

        # Allow 0 as background, but reject regions with unrelated colors.
        if not region_colors.issubset(output_colors.union({0})):
            continue

        candidate_items = [
            {
                "name": "raw_region",
                "grid": raw_grid,
            }
        ]

        candidate_items.extend(
            generate_clean_pattern_candidates(
                raw_grid,
                output_grid,
            )
        )

        for item in candidate_items:
            candidate_name = item["name"]
            candidate_grid = item["grid"]

            if candidate_grid is None:
                continue

            if len(candidate_grid) != output_h or len(candidate_grid[0]) != output_w:
                continue

            if not is_color_compatible(candidate_grid, output_grid):
                continue

            total_score = score_region_candidate(
                candidate_grid=candidate_grid,
                output_grid=output_grid,
                region_grid=raw_grid,
                candidate_name=candidate_name,
            )

            exact = candidate_grid == output_grid

            if total_score > best_score:
                best_score = total_score
                best = {
                    "strategy": "region_rule",
                    "predicted": candidate_grid,
                    "score": total_score,
                    "exact": exact,
                    "region": region,
                    "candidate": candidate_name,
                    "transform": candidate_name,
                    "train_only": True,
                    "note": (
                        "region_rule is pair-level discovery; "
                        "do leave-one-out before trusting as task-level learning"
                    ),
                }

    return best