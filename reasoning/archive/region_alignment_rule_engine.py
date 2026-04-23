from core.scoring import score_prediction


def colors(grid):
    return set(v for row in grid for v in row)


def trim_zero_margins(grid):
    if not grid or not grid[0]:
        return grid

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


def trim_uniform_border(grid):
    if not grid or not grid[0]:
        return grid

    changed = True
    current = [row[:] for row in grid]

    while changed:
        changed = False
        h = len(current)
        w = len(current[0]) if h else 0

        if h <= 1 or w <= 1:
            break

        top_row = current[0]
        bottom_row = current[-1]
        left_col = [current[r][0] for r in range(h)]
        right_col = [current[r][-1] for r in range(h)]

        if len(set(top_row)) == 1:
            current = current[1:]
            changed = True
            continue

        if len(set(bottom_row)) == 1:
            current = current[:-1]
            changed = True
            continue

        if len(set(left_col)) == 1:
            current = [row[1:] for row in current]
            changed = True
            continue

        if len(set(right_col)) == 1:
            current = [row[:-1] for row in current]
            changed = True
            continue

    return current if current else [[0]]


def generate_candidate_regions(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    regions = []

    for h in range(1, rows + 1):
        for w in range(1, cols + 1):
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


def solve_pair_region_alignment_rule(input_grid, output_grid):
    candidates = generate_candidate_regions(input_grid)

    best = None
    best_score = -10**9

    output_h = len(output_grid)
    output_w = len(output_grid[0]) if output_h else 0
    output_colors = colors(output_grid)

    for region in candidates:
        raw = region["grid"]

        variants = [
            ("raw", raw),
            ("zero_trim", trim_zero_margins(raw)),
            ("border_trim", trim_uniform_border(raw)),
            ("zero_then_border", trim_uniform_border(trim_zero_margins(raw))),
            ("border_then_zero", trim_zero_margins(trim_uniform_border(raw))),
        ]

        for variant_name, grid in variants:
            if not grid or not grid[0]:
                continue

            if len(grid) != output_h or len(grid[0]) != output_w:
                continue

            region_colors = colors(grid)
            if not region_colors.issubset(output_colors.union({0})):
                continue

            score = score_prediction(grid, output_grid)

            if score > best_score:
                best_score = score
                best = {
                    "strategy": "region_alignment_rule",
                    "predicted": grid,
                    "score": score,
                    "exact": grid == output_grid,
                    "region": region,
                    "variant": variant_name,
                }

    return best