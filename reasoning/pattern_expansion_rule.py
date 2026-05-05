# reasoning/pattern_expansion_rule.py


def grid_shape(grid):
    if grid is None:
        return 0, 0
    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def score_grid(predicted, expected):
    if predicted is None or expected is None:
        return 0

    score = 0
    for r in range(min(len(predicted), len(expected))):
        for c in range(min(len(predicted[0]), len(expected[0]))):
            if predicted[r][c] == expected[r][c]:
                score += 1

    if predicted == expected:
        score += 1000000

    return score


def make_raw_tile(input_grid, out_h, out_w):
    in_h, in_w = grid_shape(input_grid)

    return [
        [input_grid[r % in_h][c % in_w] for c in range(out_w)]
        for r in range(out_h)
    ]


def make_shifted_tile(input_grid, out_h, out_w, row_shift=0, col_shift=0):
    in_h, in_w = grid_shape(input_grid)

    pred = []
    for r in range(out_h):
        row = []
        for c in range(out_w):
            src_r = (r + c * col_shift) % in_h
            src_c = (c + r * row_shift) % in_w
            row.append(input_grid[src_r][src_c])
        pred.append(row)

    return pred


def copy_seed_top_left(pred, input_grid):
    in_h, in_w = grid_shape(input_grid)
    out_h, out_w = grid_shape(pred)

    for r in range(min(in_h, out_h)):
        for c in range(min(in_w, out_w)):
            pred[r][c] = input_grid[r][c]


def make_row_continuation(input_grid, out_h, out_w):
    """
    Keep input as top-left seed.
    Extend each known input row to the right using that row.
    Extend lower rows by repeating row behavior.
    """
    in_h, in_w = grid_shape(input_grid)
    pred = [[0 for _ in range(out_w)] for _ in range(out_h)]

    for r in range(out_h):
        src_row = input_grid[r % in_h]
        for c in range(out_w):
            pred[r][c] = src_row[c % in_w]

    copy_seed_top_left(pred, input_grid)
    return pred


def make_col_continuation(input_grid, out_h, out_w):
    """
    Keep input as top-left seed.
    Extend columns downward using that column pattern.
    """
    in_h, in_w = grid_shape(input_grid)
    pred = [[0 for _ in range(out_w)] for _ in range(out_h)]

    for r in range(out_h):
        for c in range(out_w):
            pred[r][c] = input_grid[r % in_h][c % in_w]

    copy_seed_top_left(pred, input_grid)
    return pred


def flip_binary_colors(grid):
    """
    Only flips if grid uses exactly two colors.
    Example: 7 <-> 8, or 0 <-> 3.
    """
    colors = sorted(set(v for row in grid for v in row))
    if len(colors) != 2:
        return None

    a, b = colors
    return [[b if v == a else a for v in row] for row in grid]


def make_quadrant_variants(input_grid, out_h, out_w):
    """
    Probe idea:
    build raw tile, then alter right/bottom/bottom-right areas.
    """
    base = make_raw_tile(input_grid, out_h, out_w)
    flipped = flip_binary_colors(base)

    if flipped is None:
        return []

    variants = []

    # right half flip
    pred = [row[:] for row in base]
    for r in range(out_h):
        for c in range(out_w // 2, out_w):
            pred[r][c] = flipped[r][c]
    copy_seed_top_left(pred, input_grid)
    variants.append(("right_half_flip", pred))

    # bottom half flip
    pred = [row[:] for row in base]
    for r in range(out_h // 2, out_h):
        for c in range(out_w):
            pred[r][c] = flipped[r][c]
    copy_seed_top_left(pred, input_grid)
    variants.append(("bottom_half_flip", pred))

    # bottom-right flip
    pred = [row[:] for row in base]
    for r in range(out_h // 2, out_h):
        for c in range(out_w // 2, out_w):
            pred[r][c] = flipped[r][c]
    copy_seed_top_left(pred, input_grid)
    variants.append(("bottom_right_flip", pred))

    return variants


def make_seed_completion(input_grid, out_h, out_w):
    """
    Main hypothesis for 269e22fb:
    input is a seed, output is a 20x20 completion.
    This version keeps seed fixed and tries simple continuation.
    """
    base = make_raw_tile(input_grid, out_h, out_w)
    pred = [row[:] for row in base]

    copy_seed_top_left(pred, input_grid)
    return pred


def build_candidates(input_grid, out_h, out_w):
    candidates = []

    candidates.append((
        "neighbor_propagation",
        make_neighbor_propagation(input_grid, out_h, out_w)
    ))
    candidates.append(("raw_tile", make_raw_tile(input_grid, out_h, out_w)))
    candidates.append(("seed_completion", make_seed_completion(input_grid, out_h, out_w)))
    candidates.append(("row_continuation", make_row_continuation(input_grid, out_h, out_w)))
    candidates.append(("col_continuation", make_col_continuation(input_grid, out_h, out_w)))

    for row_shift in range(-4, 5):
        for col_shift in range(-4, 5):
            if row_shift == 0 and col_shift == 0:
                continue

            pred = make_shifted_tile(
                input_grid,
                out_h,
                out_w,
                row_shift=row_shift,
                col_shift=col_shift,
            )

            copy_seed_top_left(pred, input_grid)

            candidates.append((
                f"shifted_row_{row_shift}_col_{col_shift}",
                pred,
            ))

    candidates.extend(make_quadrant_variants(input_grid, out_h, out_w))

    return candidates


def solve_pair_pattern_expansion(input_grid, output_grid):
    if input_grid is None:
        return None

    if output_grid is not None:
        out_h, out_w = grid_shape(output_grid)
    else:
        out_h, out_w = 20, 20

    best_mode = None
    best_pred = None
    best_score = -1

    for mode, pred in build_candidates(input_grid, out_h, out_w):
        score = score_grid(pred, output_grid)

        if score > best_score:
            best_score = score
            best_mode = mode
            best_pred = pred

    return {
        "strategy": "pattern_expansion_rule",
        "mode": best_mode,
        "predicted": best_pred,
        "score": best_score,
        "exact": best_pred == output_grid if output_grid is not None else False,
    }


def make_neighbor_propagation(input_grid, out_h, out_w, steps=50):
    """
    Expand grid using neighbor influence.

    Idea:
    - Start with tiled base
    - Repeatedly update cells based on neighbors
    """

    base = make_raw_tile(input_grid, out_h, out_w)
    pred = [row[:] for row in base]

    directions = [(-1,0),(1,0),(0,-1),(0,1)]

    for _ in range(steps):
        new_grid = [row[:] for row in pred]

        for r in range(out_h):
            for c in range(out_w):
                counts = {}

                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < out_h and 0 <= nc < out_w:
                        val = pred[nr][nc]
                        counts[val] = counts.get(val, 0) + 1

                if counts:
                    # take most common neighbor color
                    best = max(counts, key=counts.get)
                    new_grid[r][c] = best

        pred = new_grid

    copy_seed_top_left(pred, input_grid)
    return pred


def apply_pattern_expansion_mode(input_grid, mode, out_h=20, out_w=20):
    """
    Apply a specific mode directly (NO scoring, NO searching).

    This is the missing link for task-level learning.
    """

    candidates = build_candidates(input_grid, out_h, out_w)

    for candidate_mode, pred in candidates:
        if candidate_mode == mode:
            return pred

    # fallback if mode not found
    return None