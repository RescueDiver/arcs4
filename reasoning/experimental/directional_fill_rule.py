# reasoning/directional_fill_rule.py


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
        score += 10000

    return score


def make_raw_tile(input_grid, out_h, out_w):
    in_h, in_w = grid_shape(input_grid)

    return [
        [input_grid[r % in_h][c % in_w] for c in range(out_w)]
        for r in range(out_h)
    ]


def fill_left_to_right(grid):
    new = [row[:] for row in grid]
    h, w = grid_shape(new)

    for r in range(h):
        for c in range(1, w):
            if new[r][c] == 0:
                new[r][c] = new[r][c - 1]

    return new


def fill_right_to_left(grid):
    new = [row[:] for row in grid]
    h, w = grid_shape(new)

    for r in range(h):
        for c in range(w - 2, -1, -1):
            if new[r][c] == 0:
                new[r][c] = new[r][c + 1]

    return new


def fill_top_to_bottom(grid):
    new = [row[:] for row in grid]
    h, w = grid_shape(new)

    for r in range(1, h):
        for c in range(w):
            if new[r][c] == 0:
                new[r][c] = new[r - 1][c]

    return new


def fill_bottom_to_top(grid):
    new = [row[:] for row in grid]
    h, w = grid_shape(new)

    for r in range(h - 2, -1, -1):
        for c in range(w):
            if new[r][c] == 0:
                new[r][c] = new[r + 1][c]

    return new


def invert_zero_nonzero(grid):
    """
    For binary-style tasks:
    If grid has 0 and one nonzero color, flip 0/nonzero.
    """
    colors = sorted(set(v for row in grid for v in row))
    nonzero = [c for c in colors if c != 0]

    if len(nonzero) != 1:
        return grid

    fg = nonzero[0]
    return [[fg if v == 0 else 0 for v in row] for row in grid]


def build_candidates(input_grid, out_h, out_w):
    base = make_raw_tile(input_grid, out_h, out_w)

    candidates = []

    candidates.append(("raw_tile", base))
    candidates.append(("left_to_right", fill_left_to_right(base)))
    candidates.append(("right_to_left", fill_right_to_left(base)))
    candidates.append(("top_to_bottom", fill_top_to_bottom(base)))
    candidates.append(("bottom_to_top", fill_bottom_to_top(base)))

    # Combined directional passes
    candidates.append(("left_to_right_then_top_to_bottom", fill_top_to_bottom(fill_left_to_right(base))))
    candidates.append(("top_to_bottom_then_left_to_right", fill_left_to_right(fill_top_to_bottom(base))))
    candidates.append(("right_to_left_then_bottom_to_top", fill_bottom_to_top(fill_right_to_left(base))))
    candidates.append(("bottom_to_top_then_right_to_left", fill_right_to_left(fill_bottom_to_top(base))))

    # Inverted candidates for 0/nonzero tasks
    inverted = invert_zero_nonzero(base)
    if inverted != base:
        candidates.append(("inverted_raw_tile", inverted))
        candidates.append(("inverted_left_to_right", fill_left_to_right(inverted)))
        candidates.append(("inverted_top_to_bottom", fill_top_to_bottom(inverted)))

    return candidates


def solve_pair_directional_fill_rule(input_grid, output_grid):
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
        "strategy": "directional_fill_rule",
        "mode": best_mode,
        "predicted": best_pred,
        "score": best_score,
        "exact": best_pred == output_grid if output_grid is not None else False,
    }