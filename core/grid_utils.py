def print_grid(grid, title="GRID"):
    print(f"\n{title} (h={len(grid)}, w={len(grid[0]) if grid else 0})")
    for row in grid:
        print(" ".join(str(x) for x in row))


def crop_to_bbox(grid, bbox):
    min_r = bbox["min_r"]
    max_r = bbox["max_r"]
    min_c = bbox["min_c"]
    max_c = bbox["max_c"]

    return [row[min_c:max_c + 1] for row in grid[min_r:max_r + 1]]


def count_nonzero_cells(grid):
    return sum(1 for row in grid for v in row if v != 0)


def grids_equal(a, b):
    return a == b


def count_matches(a, b):
    if a is None or b is None:
        return 0

    rows = min(len(a), len(b))
    cols = min(len(a[0]), len(b[0])) if rows else 0

    score = 0
    for r in range(rows):
        for c in range(cols):
            if a[r][c] == b[r][c]:
                score += 1
    return score


def get_object_border_contacts(grid, obj):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    contacts = {
        "top": set(),
        "bottom": set(),
        "left": set(),
        "right": set(),
    }

    cells = set(obj["cells"])

    for r, c in cells:

        # TOP edge
        if (r - 1, c) not in cells:
            if r - 1 >= 0:
                val = grid[r - 1][c]
                if val != 0:
                    contacts["top"].add(val)

        # BOTTOM edge
        if (r + 1, c) not in cells:
            if r + 1 < rows:
                val = grid[r + 1][c]
                if val != 0:
                    contacts["bottom"].add(val)

        # LEFT edge
        if (r, c - 1) not in cells:
            if c - 1 >= 0:
                val = grid[r][c - 1]
                if val != 0:
                    contacts["left"].add(val)

        # RIGHT edge
        if (r, c + 1) not in cells:
            if c + 1 < cols:
                val = grid[r][c + 1]
                if val != 0:
                    contacts["right"].add(val)

    return contacts