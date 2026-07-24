import json
import os


TARGET_TASK_ID = "269e22fb"
TARGET_EXPECTED_PAIR = 5

# Lower = finds more possible partial matches.
# Higher = stricter.
MIN_MATCH_RATIO = 0.90

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def grid_shape(grid):
    if grid is None:
        return 0, 0
    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def load_task(task_id):
    exact_filename = f"{task_id}.json"

    candidate_paths = [
        os.path.join(BASE_DIR, "data_failures", "extracted_tasks", exact_filename),
        os.path.join(BASE_DIR, "data_failures", exact_filename),
        os.path.join(BASE_DIR, "data", exact_filename),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            if task_id in raw:
                return raw[task_id]

            if "train" in raw:
                return raw

    raise FileNotFoundError(f"Could not find task {task_id}")


def rotate_grid_90(grid):
    return [list(row) for row in zip(*grid[::-1])]


def rotate_grid_180(grid):
    return rotate_grid_90(rotate_grid_90(grid))


def rotate_grid_270(grid):
    return rotate_grid_90(rotate_grid_180(grid))


def flip_grid_horizontal(grid):
    return [row[::-1] for row in grid]


def flip_grid_vertical(grid):
    return grid[::-1]


def get_grid_transforms(grid):
    return [
        ("identity", grid),
        ("rotate_90", rotate_grid_90(grid)),
        ("rotate_180", rotate_grid_180(grid)),
        ("rotate_270", rotate_grid_270(grid)),
        ("flip_horizontal", flip_grid_horizontal(grid)),
        ("flip_vertical", flip_grid_vertical(grid)),
    ]


def count_nonzero(grid):
    return sum(1 for row in grid for v in row if v != 0)

def grid_shape(grid):
    if grid is None:
        return 0, 0

    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def score_partial_shape_match(seed_grid, expected_grid, top, left):
    """
    Partial match with consistent color mapping.

    Allows:
    - colors to change
    - rotations/flips
    - partial matching

    Rejects:
    - random filled regions
    - inconsistent recoloring
    """

    seed_h, seed_w = grid_shape(seed_grid)
    exp_h, exp_w = grid_shape(expected_grid)

    color_map = {}
    reverse_map = {}

    checked = 0
    matched = 0

    for r in range(seed_h):
        for c in range(seed_w):
            sr = seed_grid[r][c]

            rr = top + r
            cc = left + c

            if not (0 <= rr < exp_h and 0 <= cc < exp_w):
                continue

            ev = expected_grid[rr][cc]

            # Background must stay background.
            if sr == 0:
                checked += 1
                if ev == 0:
                    matched += 1
                continue

            # Seed foreground should map to expected foreground.
            if ev == 0:
                checked += 1
                continue

            checked += 1

            if sr in color_map:
                if color_map[sr] == ev:
                    matched += 1
            else:
                if ev in reverse_map:
                    continue

                color_map[sr] = ev
                reverse_map[ev] = sr
                matched += 1

    if checked == 0:
        return 0.0, 0, 0

    return matched / checked, matched, checked


def find_partial_seed_matches(seed_grid, expected_grid):
    matches = []

    expected_h, expected_w = grid_shape(expected_grid)

    for transform_name, transformed_seed in get_grid_transforms(seed_grid):
        seed_h, seed_w = grid_shape(transformed_seed)

        # Allow placement where seed is fully inside expected.
        # Later we can allow off-board partials too.
        for top in range(expected_h - seed_h + 1):
            for left in range(expected_w - seed_w + 1):
                ratio, matched, total = score_partial_shape_match(
                    transformed_seed,
                    expected_grid,
                    top,
                    left,
                )

                if ratio >= MIN_MATCH_RATIO:
                    matches.append({
                        "transform": transform_name,
                        "top": top,
                        "left": left,
                        "height": seed_h,
                        "width": seed_w,
                        "ratio": ratio,
                        "matched": matched,
                        "total": total,
                    })

    matches.sort(
        key=lambda m: (
            m["ratio"],
            m["matched"],
        ),
        reverse=True,
    )

    return matches


def main():
    task = load_task(TARGET_TASK_ID)
    train_pairs = task["train"]

    expected_grid = train_pairs[TARGET_EXPECTED_PAIR - 1]["output"]

    print("=" * 60)
    print("ONE QUESTION TEST — PARTIAL SEED SEARCH")
    print(f"Task: {TARGET_TASK_ID}")
    print(f"Question: Can expected train pair {TARGET_EXPECTED_PAIR} find the other seeds?")
    print(f"Minimum match ratio: {MIN_MATCH_RATIO}")
    print("=" * 60)

    found_any_other_seed = False

    for seed_index, pair in enumerate(train_pairs, start=1):
        if seed_index == TARGET_EXPECTED_PAIR:
            continue

        seed_grid = pair["input"]
        matches = find_partial_seed_matches(seed_grid, expected_grid)

        print()
        print(
            f"Seed {seed_index} inside Expected {TARGET_EXPECTED_PAIR}: "
            f"{len(matches)} strong partial match(es)"
        )

        for match in matches[:10]:
            percent = match["ratio"] * 100
            print(
                f"  transform={match['transform']} "
                f"top={match['top']} "
                f"left={match['left']} "
                f"size={match['height']}x{match['width']} "
                f"match={percent:.1f}% "
                f"cells={match['matched']}/{match['total']}"
            )

        if len(matches) > 10:
            print(f"  ... {len(matches) - 10} more hidden")

        if matches:
            found_any_other_seed = True

    print()
    print("=" * 60)

    if found_any_other_seed:
        print("ANSWER: YES — train pair 5 expected contains strong partial matches to other seeds.")
    else:
        print("ANSWER: NO — no strong partial matches found.")

    print("=" * 60)


if __name__ == "__main__":
    main()