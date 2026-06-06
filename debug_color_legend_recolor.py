# debug_color_legend_recolor.py

import json
import os

from core.scoring import score_prediction

from reasoning.color_legend_recolor_rule import (
    discover_color_legend_recolor_rule_for_task,
    apply_color_legend_recolor_rule,
    leave_one_out_color_legend_recolor_rule,
    find_nonzero_components,
    find_divider_panels_auto,
    select_source_block,
)


TASK_PATH = os.path.join(
    "data_failures",
    "extracted_tasks",
    "b0039139.json",
)


def print_grid(title, grid):
    print()
    print(title)

    if grid is None:
        print("None")
        return

    print(f"h={len(grid)} w={len(grid[0]) if grid else 0}")

    for row in grid:
        print(" ".join(str(v) for v in row))


def unwrap_task(raw):
    """
    Supports both:
        {"train": [...], "test": [...]}

    and:
        {"b0039139": {"train": [...], "test": [...]}}
    """

    if isinstance(raw, dict) and "train" in raw:
        return raw

    if isinstance(raw, dict):
        first_key = list(raw.keys())[0]
        return raw[first_key]

    raise ValueError("Could not unwrap ARC task.")


def print_block_debug(train_pairs):
    print()
    print("=" * 60)
    print("PANEL DEBUG")
    print("=" * 60)

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]

        panels = find_divider_panels_auto(input_grid)

        if not panels:
            panels = find_nonzero_components(input_grid)

        print()
        print(f"PAIR {pair_index + 1}")
        print(f"Panel/block count: {len(panels)}")

        for block_index, block in enumerate(panels):
            bbox = block["bbox"]

            colors = sorted(
                input_grid[r][c]
                for r, c in block["cells"]
            )

            unique_colors = sorted(set(colors))

            panel_left = block.get("panel_left")
            panel_right = block.get("panel_right")

            print(
                f"  block {block_index}: "
                f"top={bbox['top']} left={bbox['left']} "
                f"h={bbox['height']} w={bbox['width']} "
                f"panel_left={panel_left} panel_right={panel_right} "
                f"colors={unique_colors} "
                f"area={len(block['cells'])}"
            )


def print_source_color_debug(train_pairs):
    print()
    print("=" * 60)
    print("SOURCE COLOR DEBUG")
    print("=" * 60)

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        expected = pair["output"]

        blocks = find_nonzero_components(input_grid)
        source = select_source_block(blocks, input_grid)

        print()
        print(f"PAIR {pair_index + 1}")

        if source is None:
            print("No source block found.")
            continue

        bbox = source["bbox"]

        print(
            f"Source bbox: "
            f"top={bbox['top']} left={bbox['left']} "
            f"h={bbox['height']} w={bbox['width']}"
        )

        source_colors = sorted(set(
            input_grid[r][c]
            for r, c in source["cells"]
            if input_grid[r][c] != 0
        ))

        print(f"Source colors: {source_colors}")

        expected_colors = sorted(set(
            value
            for row in expected
            for value in row
            if value != 0
        ))

        print(f"Expected nonzero colors: {expected_colors}")

        expected_h = len(expected)
        expected_w = len(expected[0]) if expected else 0

        for color in source_colors:
            cells = [
                (r, c)
                for r, c in source["cells"]
                if input_grid[r][c] == color
            ]

            if not cells:
                continue

            rows = [r for r, c in cells]
            cols = [c for r, c in cells]

            tight_h = max(rows) - min(rows) + 1
            tight_w = max(cols) - min(cols) + 1

            shape_match = tight_h == expected_h and tight_w == expected_w

            print(
                f"  color {color}: "
                f"cell_count={len(cells)} "
                f"tight_shape={tight_h}x{tight_w} "
                f"matches_expected_shape={shape_match}"
            )

        print(f"Expected shape: {expected_h}x{expected_w}")


def main():
    with open(TASK_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    task = unwrap_task(raw)

    train_pairs = task["train"]
    test_pairs = task.get("test", [])

    print_block_debug(train_pairs)
    print_source_color_debug(train_pairs)

    print()
    print("=" * 60)
    print("COLOR LEGEND RECOLOR DEBUG")
    print("=" * 60)

    rule = discover_color_legend_recolor_rule_for_task(
        train_pairs,
        debug=True,
    )

    print()
    print("DISCOVERED RULE:")
    print(rule)

    if rule is None:
        print("No rule discovered.")
        return

    print()
    print("=" * 60)
    print("TRAIN CHECK")
    print("=" * 60)

    right = 0
    wrong = 0

    for idx, pair in enumerate(train_pairs):
        predicted = apply_color_legend_recolor_rule(
            task_rule=rule,
            input_grid=pair["input"],
        )

        expected = pair["output"]
        exact = predicted == expected
        score = score_prediction(predicted, expected)

        if exact:
            right += 1
        else:
            wrong += 1

        print()
        print(f"TRAIN PAIR {idx + 1}")
        print(f"Exact: {exact}")
        print(f"Score: {score}")

        if not exact:
            print_grid("EXPECTED", expected)
            print_grid("PREDICTED", predicted)

    print()
    print(f"Train right: {right}")
    print(f"Train wrong: {wrong}")

    print()
    print("=" * 60)
    print("LEAVE ONE OUT CHECK")
    print("=" * 60)

    loo = leave_one_out_color_legend_recolor_rule(train_pairs)
    print(loo)

    print()
    print("=" * 60)
    print("TEST PREDICTIONS")
    print("=" * 60)

    for idx, pair in enumerate(test_pairs):
        predicted = apply_color_legend_recolor_rule(
            task_rule=rule,
            input_grid=pair["input"],
        )

        print_grid(f"TEST {idx + 1} PREDICTED", predicted)


if __name__ == "__main__":
    main()