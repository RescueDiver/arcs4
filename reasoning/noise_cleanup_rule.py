from core.scoring import score_prediction


def solve_pair_noise_cleanup(input_grid, output_grid):
    h = len(input_grid)
    w = len(input_grid[0])

    # Step 1: remove 7
    cleaned = [
        [cell if cell != 7 else 0 for cell in row]
        for row in input_grid
    ]

    # Step 2: fill gaps
    result = [row[:] for row in cleaned]

    for r in range(h):
        for c in range(w):
            if result[r][c] == 0:
                neighbors = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if cleaned[nr][nc] in [1, 4]:
                            neighbors.append(cleaned[nr][nc])

                if neighbors:
                    result[r][c] = max(set(neighbors), key=neighbors.count)

    # Step 3: crop
    rows = [r for r in range(h) for c in range(w) if result[r][c] != 0]
    cols = [c for r in range(h) for c in range(w) if result[r][c] != 0]

    if not rows or not cols:
        return None

    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    cropped = [
        row[min_c:max_c+1]
        for row in result[min_r:max_r+1]
    ]

    # ✅ RETURN IN STANDARD FORMAT
    return {
        "predicted": cropped,
        "score": score_prediction(cropped, output_grid) if output_grid else -10**9,
        "exact": cropped == output_grid if output_grid else False
    }