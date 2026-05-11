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


def most_common_nonzero_color(grid):
    counts = Counter(v for row in grid for v in row if v != 0)

    if not counts:
        return 0

    return counts.most_common(1)[0][0]


def ordered_nonzero_colors_by_count(grid):
    counts = Counter(v for row in grid for v in row if v != 0)

    return [color for color, count in counts.most_common()]


# ============================================================
# REGION GENERATION
# ============================================================

def generate_candidate_regions(grid):
    """
    Generate every possible sub-rectangle except the whole grid.

    This is brute-force, but it is useful for small ARC grids.
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

    Useful when the raw crop has the right structure but the two active
    colors are inverted.
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

    This matches clean nested-frame tasks.
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

        # Top and bottom rows of this frame.
        for c in range(left, right + 1):
            out[top][c] = color
            out[bottom][c] = color

        # Left and right columns of this frame.
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

    This handles near-misses where a pure recursive frame puts the
    border color in the center, but expected wants the fill color.
    """
    out = recursive_frame_pattern(
        height=height,
        width=width,
        color_a=color_a,
        color_b=color_b,
    )

    if out is None:
        return None

    # Only open the true center when both dimensions are odd.
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

    This helps with outputs where the real pattern is a clean square/box
    inside a wider or taller output.
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
    Same as recursive_square_frame_in_canvas, but opens the center cell
    of the square when the square has a true center.
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

    extension_mode options:

        "fill_only"
            Right extension is all color_b.

        "right_border"
            Right extension is color_b, but the far-right column and
            top/bottom rows are color_a.

        "top_bottom_only"
            Right extension is color_b, but top/bottom rows are color_a.
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

    # Copy the recursive frame into the left side.
    for r in range(height):
        for c in range(left_width):
            out[r][c] = left_part[r][c]

    # Fill the extension.
    for r in range(height):
        for c in range(left_width, width):
            out[r][c] = color_b

    # Optional top/bottom border across the extension.
    if extension_mode in {"right_border", "top_bottom_only"}:
        for c in range(left_width, width):
            out[0][c] = color_a
            out[height - 1][c] = color_a

    # Optional far-right border.
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
    Same as left_recursive_frame_with_right_extension, but opens the center
    cell of the left recursive frame when possible.
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

    # Copy the recursive frame into the left side.
    for r in range(height):
        for c in range(left_width):
            out[r][c] = left_part[r][c]

    # Fill the extension.
    for r in range(height):
        for c in range(left_width, width):
            out[r][c] = color_b

    # Optional top/bottom border across the extension.
    if extension_mode in {"right_border", "top_bottom_only"}:
        for c in range(left_width, width):
            out[0][c] = color_a
            out[height - 1][c] = color_a

    # Optional far-right border.
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

    This is aimed at cases like 2d0172a1 pair 4.

    Difference from right_border mode:
        - keeps outer top/bottom border
        - keeps far-right border
        - does NOT create an internal vertical wall in the extension
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

    # Copy left recursive frame.
    for r in range(height):
        for c in range(left_width):
            out[r][c] = left_part[r][c]

    # Right extension stays mostly open/fill color.
    for r in range(height):
        for c in range(left_width, width):
            out[r][c] = color_b

    # Top and bottom border across the whole extension.
    for c in range(left_width, width):
        out[0][c] = color_a
        out[height - 1][c] = color_a

    # Far-right border only.
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
    Left recursive frame + open right extension + one seam marker.

    This is aimed at 2d0172a1 pair 4.

    Important discovery:
        leftw_9 gives the right left-side structure, but it creates
        a bad vertical wall at the seam column.

    So this version:
        1. builds the normal recursive frame on the left
        2. opens the right extension
        3. keeps the far-right border
        4. clears the seam column
        5. puts one marker in the middle of the seam column

    For pair 4, this should clear the bad column 8 wall while keeping
    the center marker at row 4, column 8.
    """
    if height <= 0 or width <= 0:
        return None

    if left_width <= 0 or left_width >= width:
        return None

    out = [[color_b for _ in range(width)] for _ in range(height)]

    # Use normal recursive frame, not center-open.
    # Pair 4 expected the center of the left structure to stay color_a.
    left_part = recursive_frame_pattern(
        height=height,
        width=left_width,
        color_a=color_a,
        color_b=color_b,
    )

    if left_part is None:
        return None

    # Copy left recursive frame.
    for r in range(height):
        for c in range(left_width):
            out[r][c] = left_part[r][c]

    # Open right extension.
    for r in range(height):
        for c in range(left_width, width):
            out[r][c] = color_b

    # Top and bottom border across extension.
    for c in range(left_width, width):
        out[0][c] = color_a
        out[height - 1][c] = color_a

    # Far-right border.
    for r in range(height):
        out[r][width - 1] = color_a

    # --------------------------------------------------------
    # Critical part:
    # clear the seam column from the left frame.
    #
    # For leftw_9, seam_col = 8.
    # The old candidate made column 8 a full wall.
    # Expected only wants:
    #   top border
    #   bottom border
    #   one center marker
    # --------------------------------------------------------
    seam_col = left_width - 1

    if 0 <= seam_col < width:
        for r in range(1, height - 1):
            out[r][seam_col] = color_b

        if height % 2 == 1:
            center_r = height // 2
            out[center_r][seam_col] = color_a

    return out


def clean_region_by_majority_role(region_grid, color_a, color_b):
    """
    Light cleanup candidate.

    Keeps the same shape as the raw region but forces all non-zero colors
    into one of two role colors.

    This does not invent structure. It just removes unrelated color noise.
    """
    out = copy_grid(region_grid)
    allowed = {0, color_a, color_b}

    for r in range(len(out)):
        for c in range(len(out[0])):
            if out[r][c] not in allowed:
                out[r][c] = 0

    return out


def generate_possible_left_widths(height, width):
    """
    Generate reasonable left-frame widths for extension-style outputs.

    For a wide output, the real pattern is often:
        left recursive frame + right extension.

    We try a small set of widths instead of every width to avoid too much
    noise.
    """
    possible = set()

    # Common case: left square uses output height as width.
    if 1 < height < width:
        possible.add(height)

    # Sometimes the recursive box is a bit narrower.
    if 3 <= height - 2 < width:
        possible.add(height - 2)

    if 3 <= height - 1 < width:
        possible.add(height - 1)

    # Also try near the full width.
    if 3 <= width - 1:
        possible.add(width - 1)

    if 3 <= width - 2:
        possible.add(width - 2)

    # Keep only valid extension widths.
    return sorted(w for w in possible if 2 <= w < width)


def generate_clean_pattern_candidates(region_grid, output_grid):
    """
    Generate cleaned/reconstructed alternatives for a same-size region.

    The old region_rule only returned the raw crop. This adds candidates
    like recursive frames, simple cleaned borders, and left-frame/right-
    extension patterns.
    """
    candidates = []

    h, w = grid_shape(region_grid)
    out_h, out_w = grid_shape(output_grid)

    if h != out_h or w != out_w:
        return candidates

    region_colors = sorted(nonzero_colors(region_grid))
    output_colors = ordered_nonzero_colors_by_count(output_grid)

    # Prefer output colors during train-time discovery because output_grid
    # is available for scoring. Fall back to region colors if needed.
    usable_colors = output_colors[:]

    for color in region_colors:
        if color not in usable_colors:
            usable_colors.append(color)

    if len(usable_colors) < 2:
        return candidates

    # Try every ordered pair of two active colors.
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

            # 5. Recursive square frame inside wider/taller canvas.
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

                    # 7. Extra open-extension version.
                    # This keeps the far-right border but avoids building an
                    # internal wall inside the right extension.
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

                    # Extra targeted version:
                    # left recursive frame + open extension + one center marker.
                    # This is aimed at pair 4 where the extension needs a single marker,
                    # not a full internal wall.
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
    Small bonuses prefer useful clean structure without overwhelming exactness.
    """
    base_score = score_prediction(candidate_grid, output_grid)

    candidate_colors = colors(candidate_grid)

    richness_bonus = len(candidate_colors) * 2
    nonzero_bonus = count_nonzero(candidate_grid) // 8

    clean_bonus = 0

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

    Old behavior:
        search every same-size crop and return the best raw crop.

    New behavior:
        still search every same-size crop,
        but for each crop also try cleaned/generated versions:
            - swapped colors
            - simple border
            - recursive frame
            - recursive frame with center opened
            - recursive square frame
            - recursive square frame with center opened
            - left recursive frame with right extension
            - left recursive frame with center opened and right extension
            - left recursive frame with center opened and right-open extension
            - light color-role cleanup

    This helps tasks where the region size is right but the raw crop is
    noisy and the expected output is a clean generated pattern.
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

        # ----------------------------------------------------
        # Candidate 1: original raw crop.
        # ----------------------------------------------------
        candidate_items = [
            {
                "name": "raw_region",
                "grid": raw_grid,
            }
        ]

        # ----------------------------------------------------
        # Extra cleaned/generated candidates.
        # ----------------------------------------------------
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
                }

    return best