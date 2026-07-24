# debug_mask_source_search.py
import json
import os

from core.grid_utils import print_grid
from OLD.arc_visualizer import show_three_grids


MASK_COLOR = 8


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def unwrap_task(raw, task_id):
    if "train" in raw:
        return raw

    if task_id in raw:
        return raw[task_id]

    if isinstance(raw, dict) and len(raw) == 1:
        return raw[next(iter(raw))]

    raise KeyError(f"Could not find task {task_id}")


def load_task(task_id_or_path):
    base_dir = os.path.dirname(__file__)
    value = task_id_or_path.strip().strip('"')

    if value.endswith(".json") and os.path.exists(value):
        raw = load_json(value)
        task_id = os.path.splitext(os.path.basename(value))[0]
        return task_id, unwrap_task(raw, task_id)

    failure_path = os.path.join(
        base_dir,
        "data_failures",
        "extracted_tasks",
        value + ".json",
    )

    if os.path.exists(failure_path):
        raw = load_json(failure_path)
        return value, unwrap_task(raw, value)

    data_path = os.path.join(base_dir, "data", "data.json")

    if os.path.exists(data_path):
        data = load_json(data_path)
        if value in data:
            return value, data[value]

    raise FileNotFoundError(value)


def find_mask_bbox(grid, mask_color=MASK_COLOR):
    cells = []

    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == mask_color:
                cells.append((r, c))

    if not cells:
        return None

    top = min(r for r, _ in cells)
    bottom = max(r for r, _ in cells)
    left = min(c for _, c in cells)
    right = max(c for _, c in cells)

    return {
        "top": top,
        "left": left,
        "bottom": bottom,
        "right": right,
        "height": bottom - top + 1,
        "width": right - left + 1,
    }


def crop(grid, top, left, height, width):
    return [
        row[left:left + width]
        for row in grid[top:top + height]
    ]


def patch_has_mask(patch, mask_color=MASK_COLOR):
    for row in patch:
        for value in row:
            if value == mask_color:
                return True
    return False


def score_patch(candidate, expected):
    if candidate is None:
        return 0, False

    if len(candidate) != len(expected):
        return 0, False

    if candidate and expected and len(candidate[0]) != len(expected[0]):
        return 0, False

    score = 0

    for r in range(len(expected)):
        for c in range(len(expected[0])):
            if candidate[r][c] == expected[r][c]:
                score += 1

    return score, candidate == expected


def search_visible_source_patches(input_grid, expected_grid):
    input_h = len(input_grid)
    input_w = len(input_grid[0]) if input_h else 0

    out_h = len(expected_grid)
    out_w = len(expected_grid[0]) if out_h else 0

    matches = []

    for top in range(0, input_h - out_h + 1):
        for left in range(0, input_w - out_w + 1):
            candidate = crop(input_grid, top, left, out_h, out_w)

            if patch_has_mask(candidate):
                continue

            score, exact = score_patch(candidate, expected_grid)

            matches.append({
                "top": top,
                "left": left,
                "height": out_h,
                "width": out_w,
                "score": score,
                "exact": exact,
                "patch": candidate,
            })

    matches.sort(
        key=lambda item: (
            item["exact"],
            item["score"],
        ),
        reverse=True,
    )

    return matches


def main():
    task_id_or_path = input("Task id or json path: ").strip()
    task_id, task = load_task(task_id_or_path)

    train_pairs = task.get("train", [])

    print()
    print("=" * 80)
    print(f"MASK SOURCE SEARCH: {task_id}")
    print("=" * 80)

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        expected_grid = pair["output"]

        mask_bbox = find_mask_bbox(input_grid)
        matches = search_visible_source_patches(input_grid, expected_grid)

        best = matches[0] if matches else None

        print()
        print("-" * 80)
        print(f"TRAIN PAIR {pair_index + 1}")
        print(f"mask bbox: {mask_bbox}")
        print(f"expected shape: {len(expected_grid)}x{len(expected_grid[0])}")

        if best is None:
            print("No source patches found.")
            continue

        offset_top = best["top"] - mask_bbox["top"]
        offset_left = best["left"] - mask_bbox["left"]

        print()
        print("BEST SOURCE PATCH")
        print(f"source top/left : {best['top']}, {best['left']}")
        print(f"source shape    : {best['height']}x{best['width']}")
        print(f"offset from mask: row {offset_top}, col {offset_left}")
        print(f"score           : {best['score']}")
        print(f"exact           : {best['exact']}")

        exact_matches = [m for m in matches if m["exact"]]

        print()
        print(f"Exact source matches: {len(exact_matches)}")

        for idx, match in enumerate(exact_matches[:10]):
            print(
                f"  {idx + 1}. top={match['top']} "
                f"left={match['left']} "
                f"offset=({match['top'] - mask_bbox['top']}, "
                f"{match['left'] - mask_bbox['left']})"
            )

        print_grid(expected_grid, "EXPECTED")
        print_grid(best["patch"], "BEST SOURCE PATCH")

        show_three_grids(
            input_grid,
            expected_grid,
            best["patch"],
            title_a=f"{task_id} TRAIN {pair_index + 1} INPUT",
            title_b="EXPECTED",
            title_c="BEST SOURCE PATCH",
        )


if __name__ == "__main__":
    main()