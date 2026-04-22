from copy import deepcopy
from core.scoring import score_prediction


def copy_grid(grid):
    return [row[:] for row in grid]


def grid_h(grid):
    return len(grid)


def grid_w(grid):
    return len(grid[0]) if grid else 0


def crop(grid, top, left, bottom, right):
    return [row[left:right + 1] for row in grid[top:bottom + 1]]


def count_value(grid, value):
    return sum(1 for row in grid for v in row if v == value)


def unique_colors(grid):
    return set(v for row in grid for v in row)


def nonzero_bbox(grid):
    points = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v != 0]
    if not points:
        return None
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def trim_zero_margins(grid):
    if not grid or not grid[0]:
        return [[0]]

    top = 0
    bottom = len(grid) - 1
    left = 0
    right = len(grid[0]) - 1

    while top <= bottom and all(v == 0 for v in grid[top]):
        top += 1

    while bottom >= top and all(v == 0 for v in grid[bottom]):
        bottom -= 1

    while left <= right and all(grid[r][left] == 0 for r in range(top, bottom + 1)):
        left += 1

    while right >= left and all(grid[r][right] == 0 for r in range(top, bottom + 1)):
        right -= 1

    if top > bottom or left > right:
        return [[0]]

    return [row[left:right + 1] for row in grid[top:bottom + 1]]


def get_nonzero_center_region(grid, padding=0):
    """
    Gets the tight bbox around nonzero content, optionally padded.
    """
    bbox = nonzero_bbox(grid)
    if bbox is None:
        return None

    top, left, bottom, right = bbox

    top = max(0, top - padding)
    left = max(0, left - padding)
    bottom = min(grid_h(grid) - 1, bottom + padding)
    right = min(grid_w(grid) - 1, right + padding)

    return {
        "top": top,
        "left": left,
        "bottom": bottom,
        "right": right,
        "grid": crop(grid, top, left, bottom, right),
    }


def find_uniform_rectangles(grid, min_h=1, min_w=1, max_h=None, max_w=None):
    """
    Finds all rectangles filled with a single color.
    """
    h = grid_h(grid)
    w = grid_w(grid)

    if max_h is None:
        max_h = h
    if max_w is None:
        max_w = w

    found = []

    for top in range(h):
        for left in range(w):
            color = grid[top][left]

            for rh in range(min_h, min(max_h, h - top) + 1):
                for rw in range(min_w, min(max_w, w - left) + 1):
                    ok = True
                    for r in range(top, top + rh):
                        for c in range(left, left + rw):
                            if grid[r][c] != color:
                                ok = False
                                break
                        if not ok:
                            break

                    if ok:
                        found.append({
                            "top": top,
                            "left": left,
                            "height": rh,
                            "width": rw,
                            "color": color,
                        })

    return found


def keep_maximal_uniform_rectangles(rectangles):
    """
    Keeps only rectangles not strictly contained in a larger same-color rectangle.
    """
    maximal = []

    for i, a in enumerate(rectangles):
        contained = False
        a_bottom = a["top"] + a["height"] - 1
        a_right = a["left"] + a["width"] - 1

        for j, b in enumerate(rectangles):
            if i == j:
                continue
            if a["color"] != b["color"]:
                continue

            b_bottom = b["top"] + b["height"] - 1
            b_right = b["left"] + b["width"] - 1

            if (
                b["top"] <= a["top"]
                and b["left"] <= a["left"]
                and b_bottom >= a_bottom
                and b_right >= a_right
                and (b["height"] * b["width"] > a["height"] * a["width"])
            ):
                contained = True
                break

        if not contained:
            maximal.append(a)

    return maximal


def mirror_fill_from_above(region_grid, rect):
    """
    Replace the rectangle by copying rows from above using horizontal-axis reflection.
    If rect starts at row r, then replacement row 0 comes from r-1, next from r-2, etc.
    """
    out = copy_grid(region_grid)

    top = rect["top"]
    left = rect["left"]
    height = rect["height"]
    width = rect["width"]

    if top - height < 0:
        return None

    for i in range(height):
        src_r = top - 1 - i
        dst_r = top + i
        for j in range(width):
            out[dst_r][left + j] = region_grid[src_r][left + j]

    return out


def mirror_fill_from_below(region_grid, rect):
    """
    Replace the rectangle by copying rows from below using horizontal-axis reflection.
    """
    out = copy_grid(region_grid)

    top = rect["top"]
    left = rect["left"]
    height = rect["height"]
    width = rect["width"]
    bottom = top + height - 1

    if bottom + height >= grid_h(region_grid):
        return None

    for i in range(height):
        src_r = bottom + 1 + i
        dst_r = bottom - i
        for j in range(width):
            out[dst_r][left + j] = region_grid[src_r][left + j]

    return out


def extract_rect_patch(grid, rect):
    return crop(
        grid,
        rect["top"],
        rect["left"],
        rect["top"] + rect["height"] - 1,
        rect["left"] + rect["width"] - 1
    )


def horizontal_support_score(region_grid, rect):
    """
    Measures whether rows just above or below the rectangle look like likely repair sources.
    Bigger is better.
    """
    top = rect["top"]
    left = rect["left"]
    h = rect["height"]
    w = rect["width"]
    bottom = top + h - 1

    score = 0

    # Above support
    if top - h >= 0:
        for i in range(h):
            src_r = top - 1 - i
            for j in range(w):
                if region_grid[src_r][left + j] != rect["color"]:
                    score += 1

    # Below support
    if bottom + h < grid_h(region_grid):
        for i in range(h):
            src_r = bottom + 1 + i
            for j in range(w):
                if region_grid[src_r][left + j] != rect["color"]:
                    score += 1

    return score


def rank_corruption_candidates(region_grid, rectangles):
    """
    Prefer:
    - larger rectangles
    - colors that are rare-ish in the region
    - rectangles with usable mirror support above/below
    """
    color_counts = {}
    for row in region_grid:
        for v in row:
            color_counts[v] = color_counts.get(v, 0) + 1

    ranked = []

    total_cells = grid_h(region_grid) * grid_w(region_grid)

    for rect in rectangles:
        area = rect["height"] * rect["width"]
        color_freq = color_counts.get(rect["color"], 0) / max(total_cells, 1)
        support = horizontal_support_score(region_grid, rect)

        score = 0.0
        score += area * 5.0
        score += support * 1.0
        score -= color_freq * 10.0

        ranked.append((score, rect))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


def choose_output_patch_from_repaired_region(repaired_region, expected_h=None, expected_w=None):
    """
    For now:
    - if expected size is known, search for the densest / most structured patch of that size
    - otherwise return repaired region
    """
    h = grid_h(repaired_region)
    w = grid_w(repaired_region)

    if expected_h is None or expected_w is None:
        return repaired_region

    if expected_h > h or expected_w > w:
        return None

    best = None
    best_score = -10**9

    for top in range(h - expected_h + 1):
        for left in range(w - expected_w + 1):
            patch = crop(repaired_region, top, left, top + expected_h - 1, left + expected_w - 1)

            uniq = len(unique_colors(patch))
            nonzero = sum(1 for row in patch for v in row if v != 0)
            score = uniq * 10 + nonzero

            if score > best_score:
                best_score = score
                best = {
                    "top": top,
                    "left": left,
                    "patch": patch,
                    "score": score,
                }

    return best["patch"] if best else None


def solve_pair_mirror_repair_rule(input_grid, output_grid, debug=False):
    """
    Pair-level solver:
    1. get central meaningful region
    2. find uniform corruption rectangles
    3. repair by horizontal reflection
    4. search repaired region for output-sized patch
    """
    center = get_nonzero_center_region(input_grid, padding=0)
    if center is None:
        return None

    region = center["grid"]

    rects = find_uniform_rectangles(
        region,
        min_h=2,
        min_w=2,
        max_h=min(8, grid_h(region)),
        max_w=min(12, grid_w(region)),
    )
    rects = keep_maximal_uniform_rectangles(rects)

    # ignore huge full-region rectangles
    filtered = []
    for rect in rects:
        area = rect["height"] * rect["width"]
        if area <= 3:
            continue
        if rect["height"] >= grid_h(region) * 0.9 and rect["width"] >= grid_w(region) * 0.9:
            continue
        filtered.append(rect)

    if not filtered:
        return None

    ranked_rects = rank_corruption_candidates(region, filtered)

    best = None
    best_score = -10**9

    out_h = grid_h(output_grid)
    out_w = grid_w(output_grid)

    for rect_score, rect in ranked_rects[:10]:
        repairs = []

        repaired_above = mirror_fill_from_above(region, rect)
        if repaired_above is not None:
            repairs.append(("mirror_from_above", repaired_above))

        repaired_below = mirror_fill_from_below(region, rect)
        if repaired_below is not None:
            repairs.append(("mirror_from_below", repaired_below))

        for repair_name, repaired_region in repairs:
            predicted = choose_output_patch_from_repaired_region(
                repaired_region,
                expected_h=out_h,
                expected_w=out_w,
            )
            if predicted is None:
                continue

            score = score_prediction(predicted, output_grid)

            if score > best_score:
                best_score = score
                best = {
                    "strategy": "mirror_repair_rule",
                    "predicted": predicted,
                    "score": score,
                    "exact": predicted == output_grid,
                    "repair_mode": repair_name,
                    "region_bbox": {
                        "top": center["top"],
                        "left": center["left"],
                        "bottom": center["bottom"],
                        "right": center["right"],
                    },
                    "corruption_rect": rect,
                    "rect_rank_score": rect_score,
                }

    if debug and best is not None:
        print("\nDEBUG MIRROR REPAIR:")
        print("  repair_mode    :", best["repair_mode"])
        print("  region_bbox    :", best["region_bbox"])
        print("  corruption_rect:", best["corruption_rect"])
        print("  rect_rank_score:", best["rect_rank_score"])
        print("  score          :", best["score"])
        print("  exact          :", best["exact"])

    return best


def build_mirror_repair_profile(train_pairs, debug=False):
    """
    Learn a simple reusable profile from train pairs.
    We look for consistent corruption color and repair direction.
    """
    pair_results = []
    repair_modes = {}
    corruption_colors = {}

    for idx, pair in enumerate(train_pairs):
        result = solve_pair_mirror_repair_rule(pair["input"], pair["output"], debug=debug)
        if result is None:
            return None

        pair_results.append(result)

        mode = result["repair_mode"]
        color = result["corruption_rect"]["color"]

        repair_modes[mode] = repair_modes.get(mode, 0) + 1
        corruption_colors[color] = corruption_colors.get(color, 0) + 1

    best_mode = max(repair_modes, key=repair_modes.get)
    best_color = max(corruption_colors, key=corruption_colors.get)

    profile = {
        "repair_mode": best_mode,
        "corruption_color": best_color,
    }

    if debug:
        print("\n=== MIRROR REPAIR PROFILE ===")
        print(profile)

    return profile


def solve_task_mirror_repair_rule(train_pairs, test_input, debug=False):
    """
    Task-level solver for ARC:
    - build profile from train pairs
    - locate similar corruption in test input
    - repair it
    - return repaired block or best structured patch
    """
    profile = build_mirror_repair_profile(train_pairs, debug=debug)
    if profile is None:
        return None

    center = get_nonzero_center_region(test_input, padding=0)
    if center is None:
        return None

    region = center["grid"]

    rects = find_uniform_rectangles(
        region,
        min_h=2,
        min_w=2,
        max_h=min(8, grid_h(region)),
        max_w=min(12, grid_w(region)),
    )
    rects = keep_maximal_uniform_rectangles(rects)

    filtered = []
    for rect in rects:
        area = rect["height"] * rect["width"]
        if area <= 3:
            continue
        if rect["color"] != profile["corruption_color"]:
            continue
        filtered.append(rect)

    if not filtered:
        return None

    ranked_rects = rank_corruption_candidates(region, filtered)

    # Try to infer output size from train outputs
    out_hs = [grid_h(pair["output"]) for pair in train_pairs]
    out_ws = [grid_w(pair["output"]) for pair in train_pairs]

    expected_h = max(set(out_hs), key=out_hs.count)
    expected_w = max(set(out_ws), key=out_ws.count)

    for _, rect in ranked_rects:
        if profile["repair_mode"] == "mirror_from_above":
            repaired = mirror_fill_from_above(region, rect)
        else:
            repaired = mirror_fill_from_below(region, rect)

        if repaired is None:
            continue

        predicted = choose_output_patch_from_repaired_region(
            repaired,
            expected_h=expected_h,
            expected_w=expected_w,
        )
        if predicted is None:
            continue

        result = {
            "strategy": "mirror_repair_rule",
            "predicted": predicted,
            "repair_mode": profile["repair_mode"],
            "region_bbox": {
                "top": center["top"],
                "left": center["left"],
                "bottom": center["bottom"],
                "right": center["right"],
            },
            "corruption_rect": rect,
        }

        if debug:
            print("\nDEBUG TEST MIRROR REPAIR:")
            print("  repair_mode    :", result["repair_mode"])
            print("  region_bbox    :", result["region_bbox"])
            print("  corruption_rect:", result["corruption_rect"])
            print("  predicted_h    :", grid_h(predicted))
            print("  predicted_w    :", grid_w(predicted))

        return result

    return None