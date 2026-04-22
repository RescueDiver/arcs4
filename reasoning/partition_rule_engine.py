from core.scoring import score_prediction
from core.transforms import apply_transform


TRANSFORMS = [
    "identity",
    "flip_horizontal",
    "flip_vertical",
    "rotate_180",
]


def split_horizontal(grid):
    rows = len(grid)
    if rows % 2 != 0:
        return None, None

    mid = rows // 2
    top = [row[:] for row in grid[:mid]]
    bottom = [row[:] for row in grid[mid:]]
    return top, bottom


def split_vertical(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if cols % 2 != 0:
        return None, None

    mid = cols // 2
    left = [row[:mid] for row in grid]
    right = [row[mid:] for row in grid]
    return left, right


def split_quadrants(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    if rows % 2 != 0 or cols % 2 != 0:
        return None

    mid_r = rows // 2
    mid_c = cols // 2

    top_left = [row[:mid_c] for row in grid[:mid_r]]
    top_right = [row[mid_c:] for row in grid[:mid_r]]
    bottom_left = [row[:mid_c] for row in grid[mid_r:]]
    bottom_right = [row[mid_c:] for row in grid[mid_r:]]

    return {
        "top_left": top_left,
        "top_right": top_right,
        "bottom_left": bottom_left,
        "bottom_right": bottom_right,
    }


def build_candidate(predicted, split_type, source_side, transform):
    return {
        "strategy": "partition_rule",
        "split_type": split_type,
        "source_side": source_side,
        "transform": transform,
        "predicted": predicted,
    }


def score_candidate(candidate, output_grid):
    pred = candidate["predicted"]
    score = score_prediction(pred, output_grid)
    candidate["score"] = score
    candidate["exact"] = (pred == output_grid)
    return candidate


def try_candidate(predicted, split_type, source_side, transform, output_grid, best, best_score):
    candidate = build_candidate(
        predicted=predicted,
        split_type=split_type,
        source_side=source_side,
        transform=transform,
    )
    candidate = score_candidate(candidate, output_grid)

    if candidate["score"] > best_score:
        return candidate, candidate["score"]

    return best, best_score


def solve_pair_partition_rule(input_grid, output_grid):
    best = None
    best_score = -10**9

    # -------------------------
    # Horizontal halves
    # -------------------------
    top, bottom = split_horizontal(input_grid)
    if top is not None and bottom is not None:
        for transform in TRANSFORMS:
            transformed_top = apply_transform(top, transform)
            transformed_bottom = apply_transform(bottom, transform)

            best, best_score = try_candidate(
                transformed_top, "horizontal", "top", transform, output_grid, best, best_score
            )
            best, best_score = try_candidate(
                transformed_bottom, "horizontal", "bottom", transform, output_grid, best, best_score
            )

    # -------------------------
    # Vertical halves
    # -------------------------
    left, right = split_vertical(input_grid)
    if left is not None and right is not None:
        for transform in TRANSFORMS:
            transformed_left = apply_transform(left, transform)
            transformed_right = apply_transform(right, transform)

            best, best_score = try_candidate(
                transformed_left, "vertical", "left", transform, output_grid, best, best_score
            )
            best, best_score = try_candidate(
                transformed_right, "vertical", "right", transform, output_grid, best, best_score
            )

    # -------------------------
    # Quadrants
    # -------------------------
    quadrants = split_quadrants(input_grid)
    if quadrants is not None:
        for quadrant_name, quadrant_grid in quadrants.items():
            for transform in TRANSFORMS:
                transformed_quadrant = apply_transform(quadrant_grid, transform)

                best, best_score = try_candidate(
                    transformed_quadrant,
                    "quadrant",
                    quadrant_name,
                    transform,
                    output_grid,
                    best,
                    best_score,
                )

    return best