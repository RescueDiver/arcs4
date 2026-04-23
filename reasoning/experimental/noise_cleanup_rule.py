from core.scoring import score_prediction


def replace_7_with_neighbors(grid):
    """
    Replace color 7 using nearby non-7 colors.
    Very simple local cleanup:
    - look at 4-direction neighbors
    - ignore 0 and 7
    - use most common neighbor color
    - if none found, leave as 7
    """
    h = len(grid)
    w = len(grid[0])

    new_grid = [row[:] for row in grid]

    for r in range(h):
        for c in range(w):
            if grid[r][c] == 7:
                counts = {}

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr = r + dr
                    cc = c + dc

                    if 0 <= rr < h and 0 <= cc < w:
                        val = grid[rr][cc]
                        if val != 0 and val != 7:
                            counts[val] = counts.get(val, 0) + 1

                if counts:
                    best_color = max(counts, key=counts.get)
                    new_grid[r][c] = best_color

    return new_grid


def crop_nonzero_bbox(grid):
    """
    Crop to the bounding box of all nonzero cells.
    If grid is all zero, return a copy of original grid.
    """
    h = len(grid)
    w = len(grid[0])

    cells = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                cells.append((r, c))

    if not cells:
        return [row[:] for row in grid]

    min_r = min(r for r, c in cells)
    max_r = max(r for r, c in cells)
    min_c = min(c for r, c in cells)
    max_c = max(c for r, c in cells)

    cropped = []
    for r in range(min_r, max_r + 1):
        cropped.append(grid[r][min_c:max_c + 1])

    return cropped


def solve_pair_noise_cleanup(input_grid, output_grid):
    """
    Experimental cleanup rule:
    1. replace 7 using local neighbors
    2. crop to nonzero bbox
    3. score against expected output

    Returns router-style candidate dict.
    """
    if input_grid is None:
        return None

    cleaned = replace_7_with_neighbors(input_grid)
    predicted = crop_nonzero_bbox(cleaned)

    if output_grid is None:
        return {
            "predicted": predicted,
            "score": 0,
            "exact": False,
        }

    score = score_prediction(predicted, output_grid)
    exact = predicted == output_grid

    return {
        "predicted": predicted,
        "score": score,
        "exact": exact,
    }