def copy_grid(grid):
    return [row[:] for row in grid]


def rotate_90(grid):
    return [list(row) for row in zip(*grid[::-1])]


def rotate_180(grid):
    return rotate_90(rotate_90(grid))


def rotate_270(grid):
    return rotate_90(rotate_180(grid))


def flip_horizontal(grid):
    return [row[::-1] for row in grid]


def flip_vertical(grid):
    return grid[::-1]


def apply_transform(grid, transform_name):
    if transform_name == "identity":
        return copy_grid(grid)
    if transform_name == "rotate_90":
        return rotate_90(grid)
    if transform_name == "rotate_180":
        return rotate_180(grid)
    if transform_name == "rotate_270":
        return rotate_270(grid)
    if transform_name == "flip_horizontal":
        return flip_horizontal(grid)
    if transform_name == "flip_vertical":
        return flip_vertical(grid)
    if transform_name == "rotate_90_flip_horizontal":
        return flip_horizontal(rotate_90(grid))
    if transform_name == "rotate_270_flip_horizontal":
        return flip_horizontal(rotate_270(grid))

    raise ValueError(f"Unknown transform: {transform_name}")