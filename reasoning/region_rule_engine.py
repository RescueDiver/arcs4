from core.scoring import score_prediction


def colors(grid):
    return set(v for row in grid for v in row)


def count_nonzero(grid):
    return sum(1 for row in grid for v in row if v != 0)


def generate_candidate_regions(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    regions = []

    for h in range(1, rows + 1):
        for w in range(1, cols + 1):
            # skip full grid
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


def solve_pair_region_rule(input_grid, output_grid):
    candidates = generate_candidate_regions(input_grid)

    best = None
    best_score = -10**9

    output_h = len(output_grid)
    output_w = len(output_grid[0]) if output_h else 0
    output_colors = colors(output_grid)

    for region in candidates:
        grid = region["grid"]

        # must match output size exactly
        if len(grid) != output_h or len(grid[0]) != output_w:
            continue

        region_colors = colors(grid)

        # allow 0 as background, but reject regions with unrelated colors
        if not region_colors.issubset(output_colors.union({0})):
            continue

        base_score = score_prediction(grid, output_grid)

        # simple ranking bonuses
        richness_bonus = len(region_colors) * 2
        nonzero_bonus = count_nonzero(grid) // 5

        exact = grid == output_grid
        total_score = base_score + richness_bonus + nonzero_bonus

        if exact:
            total_score += 1_000_000

        if total_score > best_score:
            best_score = total_score
            best = {
                "strategy": "region_rule",
                "predicted": grid,
                "score": total_score,
                "exact": grid == output_grid,
                "region": region,
            }

    return best