import json
import os
import sys
import tkinter as tk
from collections import Counter, deque


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_TASK_PATH = (
    r"C:\Users\johne\Desktop\ARCs4\ARCs4"
    r"\data_failures\extracted_tasks\2d0172a1.json"
)

ARC_COLORS = {
    0: "#000000",  # black
    1: "#0074D9",  # blue
    2: "#FF4136",  # red
    3: "#2ECC40",  # green
    4: "#FFDC00",  # yellow
    5: "#AAAAAA",  # gray
    6: "#F012BE",  # magenta
    7: "#FF851B",  # orange
    8: "#7FDBFF",  # cyan
    9: "#870C25",  # dark red / maroon
}

LABEL_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
]


CURRENT_TRAIN_PAIRS_FOR_POPUP = []

# =============================================================================
# TASK LOADING
# =============================================================================

def load_task(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "train" in data and "test" in data:
        return data, None

    if isinstance(data, dict) and len(data) == 1:
        key = next(iter(data.keys()))
        return data[key], key

    raise ValueError("Could not understand task JSON format.")


# =============================================================================
# BASIC GRID HELPERS
# =============================================================================

def grid_shape(grid):
    if not grid:
        return 0, 0

    return len(grid), len(grid[0])


def count_colors(grid):
    counts = Counter()

    for row in grid:
        for value in row:
            counts[value] += 1

    return dict(sorted(counts.items()))


def in_bounds(grid, r, c):
    h, w = grid_shape(grid)
    return 0 <= r < h and 0 <= c < w


def get_bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]

    top = min(rows)
    left = min(cols)
    bottom = max(rows)
    right = max(cols)

    return {
        "top": top,
        "left": left,
        "bottom": bottom,
        "right": right,
        "height": bottom - top + 1,
        "width": right - left + 1,
    }


def bbox_contains(outer_bbox, inner_bbox):
    return (
        outer_bbox["top"] <= inner_bbox["top"]
        and outer_bbox["left"] <= inner_bbox["left"]
        and outer_bbox["bottom"] >= inner_bbox["bottom"]
        and outer_bbox["right"] >= inner_bbox["right"]
        and outer_bbox != inner_bbox
    )


def combined_component_bbox(components):
    cells = []

    for obj in components:
        cells.extend(obj["cells"])

    if not cells:
        return None

    return get_bbox(cells)


def select_display_candidate(analysis):
    return max(
        analysis["candidates"],
        key=lambda candidate: candidate["component_count"],
    )


# =============================================================================
# CONNECTED COMPONENT DETECTION
# =============================================================================

def find_connected_components(grid, background_color=None):
    """
    Finds same-color 4-connected foreground components.

    This does not decide blob/ring/enclosure/frame.
    It only says: here are the visible connected pieces.
    """

    h, w = grid_shape(grid)
    visited = set()
    components = []

    for r in range(h):
        for c in range(w):
            value = grid[r][c]

            if value == background_color:
                continue

            if (r, c) in visited:
                continue

            cells = []
            queue = deque()
            queue.append((r, c))
            visited.add((r, c))

            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = cr + dr
                    nc = cc + dc

                    if not in_bounds(grid, nr, nc):
                        continue

                    if (nr, nc) in visited:
                        continue

                    if grid[nr][nc] != value:
                        continue

                    visited.add((nr, nc))
                    queue.append((nr, nc))

            bbox = get_bbox(cells)

            components.append(
                {
                    "id": f"object_{len(components) + 1}",
                    "color": value,
                    "cells": cells,
                    "cell_count": len(cells),
                    "bbox": bbox,
                }
            )

    return components


def analyze_grid(grid):
    """
    Tries every color as possible background.

    No forced background.
    No forced object names.
    No forced shape labels.
    """

    color_counts = count_colors(grid)

    candidates = []

    for background in color_counts.keys():
        components = find_connected_components(
            grid,
            background_color=background,
        )

        candidates.append(
            {
                "background": background,
                "component_count": len(components),
                "components": components,
            }
        )

    return {
        "height": len(grid),
        "width": len(grid[0]) if grid else 0,
        "color_counts": color_counts,
        "candidates": candidates,
    }


# =============================================================================
# INPUT OBJECT MEMORY
# =============================================================================

def make_input_object_memory_grid(input_grid):
    """
    This is not a solved-output guess.

    It copies the visible input objects onto a clean same-size grid so we can
    visually inspect what the detector remembers from the input.
    """

    input_analysis = analyze_grid(input_grid)
    display_candidate = select_display_candidate(input_analysis)

    background = display_candidate["background"]
    components = display_candidate["components"]

    h, w = grid_shape(input_grid)

    memory_grid = [
        [background for _ in range(w)]
        for _ in range(h)
    ]

    for obj in components:
        color = obj["color"]

        for r, c in obj["cells"]:
            memory_grid[r][c] = color

    return memory_grid


def draw_learned_bbox_shape(sketch_grid, bbox, color, behavior):
    """
    Draws the learned output object location into the sketch grid.

    This does not name the object as a ring/blob/etc.
    It only uses neutral behavior:
    - became_single_cell -> draw one cell at learned output bbox
    - became_smaller -> draw the learned output bbox outline
    - fallback -> fill learned output bbox
    """

    h, w = grid_shape(sketch_grid)

    top = bbox["top"]
    left = bbox["left"]
    bottom = bbox["bottom"]
    right = bbox["right"]

    if behavior == "became_single_cell":
        rr = top
        cc = left

        if 0 <= rr < h and 0 <= cc < w:
            sketch_grid[rr][cc] = color

        return

    if behavior == "became_smaller":
        for c in range(left, right + 1):
            if 0 <= top < h and 0 <= c < w:
                sketch_grid[top][c] = color
            if 0 <= bottom < h and 0 <= c < w:
                sketch_grid[bottom][c] = color

        for r in range(top, bottom + 1):
            if 0 <= r < h and 0 <= left < w:
                sketch_grid[r][left] = color
            if 0 <= r < h and 0 <= right < w:
                sketch_grid[r][right] = color

        return

    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if 0 <= r < h and 0 <= c < w:
                sketch_grid[r][c] = color


def make_train_story_solved_sketch_grid(input_grid, output_grid):
    """
    Train-only placed visual sketch.

    This is NOT a final solver and is NOT used for scoring.
    It uses the known train output size/background and the learned train-pair
    output locations so we can inspect whether the story is being placed in
    the right visual locations.

    It still does not use forced labels like blob/ring/enclosure.
    It only uses:
    - object order from the selected component view
    - learned behavior from input object -> output object
    - learned output bbox placement from the train pair
    """

    input_analysis = analyze_grid(input_grid)
    output_analysis = analyze_grid(output_grid)

    input_candidate = select_display_candidate(input_analysis)
    expected_count = input_candidate["component_count"]

    matching_output_candidates = [
        candidate
        for candidate in output_analysis["candidates"]
        if candidate["component_count"] == expected_count
    ]

    if matching_output_candidates:
        output_candidate = matching_output_candidates[0]
    else:
        output_candidate = select_display_candidate(output_analysis)

    h, w = grid_shape(output_grid)
    background = output_candidate["background"]

    sketch_grid = [
        [background for _ in range(w)]
        for _ in range(h)
    ]

    input_objects = input_candidate["components"]
    output_objects = output_candidate["components"]
    pair_count = min(len(input_objects), len(output_objects))

    for index in range(pair_count):
        input_obj = input_objects[index]
        output_obj = output_objects[index]

        behavior = describe_object_behavior(input_obj, output_obj)
        color = output_obj["color"]
        learned_output_bbox = output_obj["bbox"]

        draw_learned_bbox_shape(
            sketch_grid=sketch_grid,
            bbox=learned_output_bbox,
            color=color,
            behavior=behavior,
        )

    return sketch_grid

def make_blind_guess_from_input(input_grid):
    """
    This is only a story-level guess from input.

    It does not make final pixels.
    It only records object count and foreground colors from the chosen input view.
    """

    input_analysis = analyze_grid(input_grid)
    display_candidate = select_display_candidate(input_analysis)

    foreground_colors = sorted(
        {
            obj["color"]
            for obj in display_candidate["components"]
        }
    )

    return {
        "input_background_used_for_guess": display_candidate["background"],
        "expected_object_count": display_candidate["component_count"],
        "expected_foreground_colors": foreground_colors,
    }


def print_size_and_placement_learning_check(train_pairs):
    print()
    print("=" * 80)
    print("SIZE / PLACEMENT / IDENTITY LEARNING CHECK")
    print("=" * 80)

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        output_grid = pair["output"]

        input_h, input_w = grid_shape(input_grid)
        output_h, output_w = grid_shape(output_grid)

        input_analysis = analyze_grid(input_grid)
        output_analysis = analyze_grid(output_grid)

        input_candidate = select_display_candidate(input_analysis)
        expected_count = input_candidate["component_count"]

        matching_output_candidates = [
            candidate
            for candidate in output_analysis["candidates"]
            if candidate["component_count"] == expected_count
        ]

        if matching_output_candidates:
            output_candidate = matching_output_candidates[0]
        else:
            output_candidate = select_display_candidate(output_analysis)

        input_objects = input_candidate["components"]
        output_objects = output_candidate["components"]

        print()
        print(f"PAIR {pair_index}")
        print("-" * 80)
        print(f"grid size: {input_h}x{input_w} -> {output_h}x{output_w}")
        print(f"input background : {input_candidate['background']}")
        print(f"output background: {output_candidate['background']}")
        print(f"objects: {len(input_objects)} -> {len(output_objects)}")

        pair_count = min(len(input_objects), len(output_objects))

        for index in range(pair_count):
            input_obj = input_objects[index]
            output_obj = output_objects[index]

            input_bbox = input_obj["bbox"]
            output_bbox = output_obj["bbox"]

            behavior = describe_object_behavior(input_obj, output_obj)

            print(
                f"  {input_obj['id']} -> {output_obj['id']} | "
                f"color {input_obj['color']}->{output_obj['color']} | "
                f"bbox "
                f"{input_bbox['height']}x{input_bbox['width']} "
                f"at top={input_bbox['top']} left={input_bbox['left']} "
                f"-> "
                f"{output_bbox['height']}x{output_bbox['width']} "
                f"at top={output_bbox['top']} left={output_bbox['left']} | "
                f"{behavior}"
            )


# =============================================================================
# CONSOLE PRINTING: COMPONENTS
# =============================================================================

def print_grid_analysis(title, grid):
    analysis = analyze_grid(grid)

    print()
    print(title)
    print("-" * 80)
    print(f"- grid size    : {analysis['height']}x{analysis['width']}")
    print(f"- color counts : {analysis['color_counts']}")

    print()
    print("BACKGROUND CANDIDATES")

    for candidate in analysis["candidates"]:
        print()
        print(f"color {candidate['background']} as background:")
        print(f"  component count: {candidate['component_count']}")

        for obj in candidate["components"]:
            bbox = obj["bbox"]
            print(
                f"    {obj['id']}: "
                f"color={obj['color']} "
                f"cells={obj['cell_count']} "
                f"bbox=top{bbox['top']},left{bbox['left']},"
                f"h{bbox['height']},w{bbox['width']}"
            )


def print_blind_guess_from_input(input_grid):
    blind_guess = make_blind_guess_from_input(input_grid)

    print()
    print("BLIND STORY GUESS FROM INPUT ONLY")
    print("-" * 80)
    print(
        "input display background: "
        f"{blind_guess['input_background_used_for_guess']}"
    )
    print(
        "expected output object count: "
        f"{blind_guess['expected_object_count']}"
    )
    print(
        "expected output foreground colors: "
        f"{blind_guess['expected_foreground_colors']}"
    )

    return blind_guess


def print_output_candidate_match_to_blind_guess(output_grid, blind_guess):
    analysis = analyze_grid(output_grid)
    expected_count = blind_guess["expected_object_count"]

    print()
    print("OUTPUT CANDIDATE MATCH TO BLIND STORY GUESS")
    print("-" * 80)

    for candidate in analysis["candidates"]:
        diff = abs(candidate["component_count"] - expected_count)

        marker = ""
        if diff == 0:
            marker = "  <-- COUNT MATCH"

        print(
            f"color {candidate['background']} as background: "
            f"objects={candidate['component_count']} "
            f"difference={diff}"
            f"{marker}"
        )


# =============================================================================
# CONSOLE PRINTING: RELATIONSHIPS
# =============================================================================

def get_relationship_facts(grid):
    analysis = analyze_grid(grid)
    display_candidate = select_display_candidate(analysis)

    objects = display_candidate["components"]

    facts = []

    for outer in objects:
        for inner in objects:
            if outer["id"] == inner["id"]:
                continue

            if bbox_contains(outer["bbox"], inner["bbox"]):
                facts.append((inner["id"], "inside_bbox", outer["id"]))

    return {
        "background": display_candidate["background"],
        "object_count": len(objects),
        "facts": facts,
    }


def print_relationship_check(title, grid):
    result = get_relationship_facts(grid)

    print()
    print(title)
    print("-" * 80)
    print(f"display background used: {result['background']}")
    print(f"object count: {result['object_count']}")

    if not result["facts"]:
        print("no bbox-inside-bbox relationships found")
        return

    for inner_id, relation, outer_id in result["facts"]:
        print(f"{inner_id} {relation} {outer_id}")


def print_relationship_preservation_check(input_grid, output_grid):
    input_result = get_relationship_facts(input_grid)
    output_result = get_relationship_facts(output_grid)

    input_facts = set(input_result["facts"])
    output_facts = set(output_result["facts"])

    preserved = sorted(input_facts & output_facts)
    missing = sorted(input_facts - output_facts)
    new = sorted(output_facts - input_facts)

    print()
    print("RELATIONSHIP PRESERVATION CHECK")
    print("-" * 80)
    print(f"input background used : {input_result['background']}")
    print(f"output background used: {output_result['background']}")
    print(f"input relationship count : {len(input_facts)}")
    print(f"output relationship count: {len(output_facts)}")
    print(f"preserved count: {len(preserved)} / {len(input_facts)}")

    print()
    print("preserved relationships:")
    if preserved:
        for inner_id, relation, outer_id in preserved:
            print(f"  {inner_id} {relation} {outer_id}")
    else:
        print("  none")

    print()
    print("missing from output:")
    if missing:
        for inner_id, relation, outer_id in missing:
            print(f"  {inner_id} {relation} {outer_id}")
    else:
        print("  none")

    print()
    print("new in output:")
    if new:
        for inner_id, relation, outer_id in new:
            print(f"  {inner_id} {relation} {outer_id}")
    else:
        print("  none")


# =============================================================================
# CONSOLE PRINTING: BEHAVIOR
# =============================================================================

def describe_object_behavior(input_obj, output_obj):
    output_bbox = output_obj["bbox"]

    if (
        output_bbox["height"] == 1
        and output_bbox["width"] == 1
        and output_obj["cell_count"] == 1
    ):
        return "became_single_cell"

    if output_obj["cell_count"] < input_obj["cell_count"]:
        return "became_smaller"

    if output_obj["cell_count"] == input_obj["cell_count"]:
        return "same_cell_count"

    if output_obj["cell_count"] > input_obj["cell_count"]:
        return "became_larger"

    return "unknown_change"


def print_object_behavior_check(input_grid, output_grid):
    input_analysis = analyze_grid(input_grid)
    output_analysis = analyze_grid(output_grid)

    input_candidate = select_display_candidate(input_analysis)
    expected_count = input_candidate["component_count"]

    matching_output_candidates = [
        candidate
        for candidate in output_analysis["candidates"]
        if candidate["component_count"] == expected_count
    ]

    print()
    print("OBJECT BEHAVIOR CHECK")
    print("-" * 80)
    print(f"input background used: {input_candidate['background']}")
    print(f"expected object count: {expected_count}")

    if not matching_output_candidates:
        print("no output candidate matched object count")
        return

    output_candidate = matching_output_candidates[0]
    print(f"matched output background: {output_candidate['background']}")

    input_objects = input_candidate["components"]
    output_objects = output_candidate["components"]

    pair_count = min(len(input_objects), len(output_objects))

    for index in range(pair_count):
        input_obj = input_objects[index]
        output_obj = output_objects[index]

        input_bbox = input_obj["bbox"]
        output_bbox = output_obj["bbox"]

        behavior = describe_object_behavior(input_obj, output_obj)

        print(
            f"{input_obj['id']} -> {output_obj['id']}: "
            f"input h={input_bbox['height']} "
            f"w={input_bbox['width']} "
            f"cells={input_obj['cell_count']} "
            f"-> "
            f"output h={output_bbox['height']} "
            f"w={output_bbox['width']} "
            f"cells={output_obj['cell_count']} "
            f"=> {behavior}"
        )


# =============================================================================
# STORY SUMMARY / TASK LEARNING
# =============================================================================

def get_story_summary(input_grid, output_grid):
    input_relationships = get_relationship_facts(input_grid)
    output_relationships = get_relationship_facts(output_grid)

    input_facts = set(input_relationships["facts"])
    output_facts = set(output_relationships["facts"])

    preserved = input_facts & output_facts

    input_analysis = analyze_grid(input_grid)
    output_analysis = analyze_grid(output_grid)

    input_candidate = select_display_candidate(input_analysis)
    expected_count = input_candidate["component_count"]

    matching_output_candidates = [
        candidate
        for candidate in output_analysis["candidates"]
        if candidate["component_count"] == expected_count
    ]

    behavior_counts = {}

    if matching_output_candidates:
        output_candidate = matching_output_candidates[0]

        input_objects = input_candidate["components"]
        output_objects = output_candidate["components"]

        pair_count = min(len(input_objects), len(output_objects))

        for index in range(pair_count):
            behavior = describe_object_behavior(
                input_objects[index],
                output_objects[index],
            )

            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1

    return {
        "input_object_count": input_relationships["object_count"],
        "output_object_count": output_relationships["object_count"],
        "input_relationship_count": len(input_facts),
        "output_relationship_count": len(output_facts),
        "preserved_relationship_count": len(preserved),
        "behavior_counts": behavior_counts,
    }


def print_story_summary_check(input_grid, output_grid):
    summary = get_story_summary(input_grid, output_grid)

    print()
    print("STORY SUMMARY CHECK")
    print("-" * 80)
    print(f"input objects : {summary['input_object_count']}")
    print(f"output objects: {summary['output_object_count']}")
    print(f"input relationships : {summary['input_relationship_count']}")
    print(f"output relationships: {summary['output_relationship_count']}")
    print(
        "preserved relationships: "
        f"{summary['preserved_relationship_count']} / "
        f"{summary['input_relationship_count']}"
    )

    print()
    print("behavior counts:")
    if summary["behavior_counts"]:
        for behavior, count in sorted(summary["behavior_counts"].items()):
            print(f"  {behavior}: {count}")
    else:
        print("  no behavior match available")


def learn_task_story(train_pairs):
    summaries = [
        get_story_summary(pair["input"], pair["output"])
        for pair in train_pairs
    ]

    object_count_preserved_all = all(
        summary["input_object_count"] == summary["output_object_count"]
        for summary in summaries
    )

    relationship_count_preserved_all = all(
        summary["input_relationship_count"] == summary["output_relationship_count"]
        for summary in summaries
    )

    every_pair_has_smaller = all(
        summary["behavior_counts"].get("became_smaller", 0) > 0
        for summary in summaries
    )

    every_pair_has_single_cell = all(
        summary["behavior_counts"].get("became_single_cell", 0) > 0
        for summary in summaries
    )

    return {
        "train_pair_count": len(summaries),
        "summaries": summaries,
        "object_count_preserved_all": object_count_preserved_all,
        "relationship_count_preserved_all": relationship_count_preserved_all,
        "every_pair_has_smaller": every_pair_has_smaller,
        "every_pair_has_single_cell": every_pair_has_single_cell,
    }


def print_task_story_learning_check(train_pairs):
    learned = learn_task_story(train_pairs)

    print()
    print("=" * 80)
    print("TASK STORY LEARNING CHECK")
    print("=" * 80)

    print(f"train pair count: {learned['train_pair_count']}")
    print()
    print(
        "object count preserved in all pairs: "
        f"{learned['object_count_preserved_all']}"
    )
    print(
        "relationship count preserved in all pairs: "
        f"{learned['relationship_count_preserved_all']}"
    )
    print(
        "every pair has became_smaller: "
        f"{learned['every_pair_has_smaller']}"
    )
    print(
        "every pair has became_single_cell: "
        f"{learned['every_pair_has_single_cell']}"
    )

    print()
    print("pair summaries:")
    for index, summary in enumerate(learned["summaries"]):
        print(
            f"pair {index}: "
            f"objects {summary['input_object_count']}->"
            f"{summary['output_object_count']}, "
            f"relationships {summary['input_relationship_count']}->"
            f"{summary['output_relationship_count']}, "
            f"preserved {summary['preserved_relationship_count']}/"
            f"{summary['input_relationship_count']}, "
            f"behaviors {summary['behavior_counts']}"
        )


def print_test_story_expectation_check(test_index, test_grid, train_pairs):
    learned = learn_task_story(train_pairs)

    test_relationships = get_relationship_facts(test_grid)
    test_facts = set(test_relationships["facts"])

    print()
    print("TEST STORY EXPECTATION CHECK")
    print("-" * 80)
    print(f"test pair: {test_index}")
    print(f"test objects: {test_relationships['object_count']}")
    print(f"test relationships: {len(test_facts)}")

    print()
    print("learned from train:")
    print(
        "  object count should be preserved: "
        f"{learned['object_count_preserved_all']}"
    )
    print(
        "  relationship count should be preserved: "
        f"{learned['relationship_count_preserved_all']}"
    )
    print(
        "  output should contain became_smaller behavior: "
        f"{learned['every_pair_has_smaller']}"
    )
    print(
        "  output should contain became_single_cell behavior: "
        f"{learned['every_pair_has_single_cell']}"
    )

    print()
    print("test expectation:")

    if learned["object_count_preserved_all"]:
        print(
            "  expected output object count: "
            f"{test_relationships['object_count']}"
        )

    if learned["relationship_count_preserved_all"]:
        print(
            "  expected output relationship count: "
            f"{len(test_facts)}"
        )

    if learned["every_pair_has_smaller"]:
        print("  expected at least one object to become smaller")

    if learned["every_pair_has_single_cell"]:
        print("  expected at least one object to become single cell")


# =============================================================================
# TEXT BLOCKS FOR POPUPS
# =============================================================================

def component_text_block(title, analysis):
    lines = []
    lines.append(title)
    lines.append("-" * 80)
    lines.append(f"grid size: {analysis['height']}x{analysis['width']}")
    lines.append(f"color counts: {analysis['color_counts']}")
    lines.append("")

    for candidate in analysis["candidates"]:
        lines.append(f"color {candidate['background']} as background:")
        lines.append(f"  object count: {candidate['component_count']}")

        for obj in candidate["components"]:
            bbox = obj["bbox"]
            lines.append(
                f"  {obj['id']}: "
                f"color={obj['color']} "
                f"cells={obj['cell_count']} "
                f"bbox=top{bbox['top']},left{bbox['left']},"
                f"h{bbox['height']},w{bbox['width']}"
            )

        lines.append("")

    return "\n".join(lines)


def blind_guess_text_block(input_grid):
    blind_guess = make_blind_guess_from_input(input_grid)

    lines = []
    lines.append("BLIND STORY GUESS FROM INPUT ONLY")
    lines.append("-" * 80)
    lines.append(
        "input background used: "
        f"{blind_guess['input_background_used_for_guess']}"
    )
    lines.append(
        "expected output object count: "
        f"{blind_guess['expected_object_count']}"
    )
    lines.append(
        "expected output foreground colors: "
        f"{blind_guess['expected_foreground_colors']}"
    )

    return "\n".join(lines)


def relationship_text_block(title, grid):
    result = get_relationship_facts(grid)

    lines = []
    lines.append(title)
    lines.append("-" * 80)
    lines.append(f"background used: {result['background']}")
    lines.append(f"object count: {result['object_count']}")
    lines.append("")

    if not result["facts"]:
        lines.append("no bbox-inside-bbox relationships found")
    else:
        for inner_id, relation, outer_id in result["facts"]:
            lines.append(f"{inner_id} {relation} {outer_id}")

    return "\n".join(lines)


def story_card_text_block(input_grid, output_grid):
    summary = get_story_summary(input_grid, output_grid)

    lines = []
    lines.append("STORY CARD - NOT SOLVED OUTPUT")
    lines.append("-" * 80)
    lines.append(
        "objects: "
        f"{summary['input_object_count']} -> "
        f"{summary['output_object_count']}"
    )
    lines.append(
        "relationships: "
        f"{summary['input_relationship_count']} -> "
        f"{summary['output_relationship_count']}"
    )
    lines.append(
        "preserved relationships: "
        f"{summary['preserved_relationship_count']} / "
        f"{summary['input_relationship_count']}"
    )
    lines.append("")
    lines.append("behavior counts:")

    if summary["behavior_counts"]:
        for behavior, count in sorted(summary["behavior_counts"].items()):
            lines.append(f"  {behavior}: {count}")
    else:
        lines.append("  no behavior match available")

    return "\n".join(lines)


def output_candidate_match_text_block(output_grid, blind_guess):
    analysis = analyze_grid(output_grid)
    expected_count = blind_guess["expected_object_count"]

    lines = []
    lines.append("OUTPUT CANDIDATE MATCH TO BLIND STORY GUESS")
    lines.append("-" * 80)

    for candidate in analysis["candidates"]:
        diff = abs(candidate["component_count"] - expected_count)

        marker = ""
        if diff == 0:
            marker = "  <-- COUNT MATCH"

        lines.append(
            f"color {candidate['background']} as background: "
            f"objects={candidate['component_count']} "
            f"difference={diff}"
            f"{marker}"
        )

    return "\n".join(lines)


def test_story_expectation_text_block(test_index, test_grid, train_pairs):
    learned = learn_task_story(train_pairs)

    test_relationships = get_relationship_facts(test_grid)
    test_facts = set(test_relationships["facts"])

    lines = []
    lines.append("TEST STORY EXPECTATION - NOT SOLVED OUTPUT")
    lines.append("-" * 80)
    lines.append(f"test pair: {test_index}")
    lines.append(f"test objects: {test_relationships['object_count']}")
    lines.append(f"test relationships: {len(test_facts)}")
    lines.append("")
    lines.append("learned from train:")
    lines.append(
        "  object count should be preserved: "
        f"{learned['object_count_preserved_all']}"
    )
    lines.append(
        "  relationship count should be preserved: "
        f"{learned['relationship_count_preserved_all']}"
    )
    lines.append(
        "  output should contain became_smaller behavior: "
        f"{learned['every_pair_has_smaller']}"
    )
    lines.append(
        "  output should contain became_single_cell behavior: "
        f"{learned['every_pair_has_single_cell']}"
    )
    lines.append("")
    lines.append("test expectation:")

    if learned["object_count_preserved_all"]:
        lines.append(
            "  expected output object count: "
            f"{test_relationships['object_count']}"
        )

    if learned["relationship_count_preserved_all"]:
        lines.append(
            "  expected output relationship count: "
            f"{len(test_facts)}"
        )

    if learned["every_pair_has_smaller"]:
        lines.append("  expected at least one object to become smaller")

    if learned["every_pair_has_single_cell"]:
        lines.append("  expected at least one object to become single cell")

    return "\n".join(lines)


# =============================================================================
# CONSOLE MAIN REPORT
# =============================================================================

def print_task_component_summary(task):
    print()
    print("=" * 80)
    print("RAW COMPONENT DEBUGGER")
    print("=" * 80)

    for pair_index, pair in enumerate(task["train"]):
        print()
        print("=" * 80)
        print(f"TRAIN PAIR {pair_index}")
        print("=" * 80)

        print_grid_analysis("INPUT COMPONENTS", pair["input"])

        blind_guess = print_blind_guess_from_input(pair["input"])

        print_relationship_check(
            "INPUT RELATIONSHIP CHECK",
            pair["input"],
        )

        print_relationship_check(
            "OUTPUT RELATIONSHIP CHECK",
            pair["output"],
        )

        print_relationship_preservation_check(
            pair["input"],
            pair["output"],
        )

        print_object_behavior_check(
            pair["input"],
            pair["output"],
        )

        print_story_summary_check(
            pair["input"],
            pair["output"],
        )

        print_grid_analysis("OUTPUT COMPONENTS", pair["output"])

        print_output_candidate_match_to_blind_guess(
            pair["output"],
            blind_guess,
        )

    print_task_story_learning_check(task["train"])
    print_size_and_placement_learning_check(task["train"])
    print_identity_role_learning_check(task["train"])

    for test_index, pair in enumerate(task.get("test", [])):
        print()
        print("=" * 80)
        print(f"TEST PAIR {test_index}")
        print("=" * 80)

        print_grid_analysis("TEST INPUT COMPONENTS", pair["input"])

        print_test_story_expectation_check(
            test_index,
            pair["input"],
            task["train"],
        )


def task_level_rule_candidates_text_block(train_pairs):
    lines = []
    lines.append("TASK-LEVEL RULE CANDIDATES - NOT OUTPUT GUESS")
    lines.append("-" * 80)

    size_facts = []
    background_same_all = True
    object_count_same_all = True
    relationship_count_same_all = True

    behavior_totals = Counter()

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        output_grid = pair["output"]

        input_h, input_w = grid_shape(input_grid)
        output_h, output_w = grid_shape(output_grid)

        input_analysis = analyze_grid(input_grid)
        output_analysis = analyze_grid(output_grid)

        input_candidate = select_display_candidate(input_analysis)
        expected_count = input_candidate["component_count"]

        matching_output_candidates = [
            candidate
            for candidate in output_analysis["candidates"]
            if candidate["component_count"] == expected_count
        ]

        if matching_output_candidates:
            output_candidate = matching_output_candidates[0]
        else:
            output_candidate = select_display_candidate(output_analysis)

        input_objects = input_candidate["components"]
        output_objects = output_candidate["components"]

        input_relationships = get_relationship_facts(input_grid)
        output_relationships = get_relationship_facts(output_grid)

        if input_candidate["background"] != output_candidate["background"]:
            background_same_all = False

        if len(input_objects) != len(output_objects):
            object_count_same_all = False

        if (
            len(input_relationships["facts"])
            != len(output_relationships["facts"])
        ):
            relationship_count_same_all = False

        size_facts.append(
            {
                "pair_index": pair_index,
                "input_grid": (input_h, input_w),
                "output_grid": (output_h, output_w),
                "largest_input_bbox": input_objects[0]["bbox"] if input_objects else None,
                "largest_output_bbox": output_objects[0]["bbox"] if output_objects else None,
            }
        )

        pair_count = min(len(input_objects), len(output_objects))

        for index in range(pair_count):
            behavior = describe_object_behavior(
                input_objects[index],
                output_objects[index],
            )
            behavior_totals[behavior] += 1

    lines.append("CONSISTENT FACTS")
    lines.append(f"background preserved all pairs: {background_same_all}")
    lines.append(f"object count preserved all pairs: {object_count_same_all}")
    lines.append(
        f"relationship count preserved all pairs: "
        f"{relationship_count_same_all}"
    )
    lines.append("")

    lines.append("BEHAVIOR TOTALS ACROSS TRAIN")
    if behavior_totals:
        for behavior, count in sorted(behavior_totals.items()):
            lines.append(f"  {behavior}: {count}")
    else:
        lines.append("  none")
    lines.append("")

    lines.append("SIZE RULE CANDIDATES")
    for fact in size_facts:
        pair_index = fact["pair_index"]
        input_h, input_w = fact["input_grid"]
        output_h, output_w = fact["output_grid"]

        lines.append(
            f"pair {pair_index}: grid {input_h}x{input_w} "
            f"-> {output_h}x{output_w}"
        )

        largest_input_bbox = fact["largest_input_bbox"]
        largest_output_bbox = fact["largest_output_bbox"]

        if largest_input_bbox and largest_output_bbox:
            lines.append(
                f"  largest bbox "
                f"{largest_input_bbox['height']}x"
                f"{largest_input_bbox['width']} "
                f"-> "
                f"{largest_output_bbox['height']}x"
                f"{largest_output_bbox['width']}"
            )

    lines.append("")
    lines.append("IDENTITY RULE CANDIDATE")
    lines.append("  current check: scan-order pairing only")
    lines.append("  warning: pair 0 already shows scan-order may be unstable")
    lines.append("")
    lines.append("NEXT LEARNING QUESTION")
    lines.append("  Should identity be matched by relationship role instead of scan order?")

    return "\n".join(lines)


def get_object_role_facts(grid):
    relationship_result = get_relationship_facts(grid)

    analysis = analyze_grid(grid)
    candidate = select_display_candidate(analysis)
    objects = candidate["components"]

    inside_map = {
        obj["id"]: []
        for obj in objects
    }

    contains_map = {
        obj["id"]: []
        for obj in objects
    }

    for inner_id, relation, outer_id in relationship_result["facts"]:
        if relation != "inside_bbox":
            continue

        if inner_id in inside_map:
            inside_map[inner_id].append(outer_id)

        if outer_id in contains_map:
            contains_map[outer_id].append(inner_id)

    role_facts = []

    for obj in objects:
        obj_id = obj["id"]
        bbox = obj["bbox"]

        role_facts.append(
            {
                "id": obj_id,
                "color": obj["color"],
                "cell_count": obj["cell_count"],
                "bbox": bbox,
                "inside_count": len(inside_map[obj_id]),
                "contains_count": len(contains_map[obj_id]),
                "inside": inside_map[obj_id],
                "contains": contains_map[obj_id],
            }
        )

    return role_facts


def print_identity_role_learning_check(train_pairs):
    print()
    print("=" * 80)
    print("IDENTITY ROLE LEARNING CHECK")
    print("=" * 80)

    for pair_index, pair in enumerate(train_pairs):
        print()
        print(f"PAIR {pair_index}")
        print("-" * 80)

        input_roles = get_object_role_facts(pair["input"])
        output_roles = get_object_role_facts(pair["output"])

        print("INPUT ROLES")
        for role in input_roles:
            print(
                f"  {role['id']}: "
                f"cells={role['cell_count']} "
                f"bbox={role['bbox']['height']}x{role['bbox']['width']} "
                f"inside_count={role['inside_count']} "
                f"contains_count={role['contains_count']} "
                f"inside={role['inside']} "
                f"contains={role['contains']}"
            )

        print()
        print("OUTPUT ROLES")
        for role in output_roles:
            print(
                f"  {role['id']}: "
                f"cells={role['cell_count']} "
                f"bbox={role['bbox']['height']}x{role['bbox']['width']} "
                f"inside_count={role['inside_count']} "
                f"contains_count={role['contains_count']} "
                f"inside={role['inside']} "
                f"contains={role['contains']}"
            )


def learner_dashboard_text_block(pair_index, input_grid, output_grid, train_pairs):
    input_analysis = analyze_grid(input_grid)
    output_analysis = analyze_grid(output_grid)

    input_candidate = select_display_candidate(input_analysis)

    expected_count = input_candidate["component_count"]

    matching_output_candidates = [
        candidate
        for candidate in output_analysis["candidates"]
        if candidate["component_count"] == expected_count
    ]

    if matching_output_candidates:
        output_candidate = matching_output_candidates[0]
    else:
        output_candidate = select_display_candidate(output_analysis)

    input_objects = input_candidate["components"]
    output_objects = output_candidate["components"]

    input_relationships = get_relationship_facts(input_grid)
    output_relationships = get_relationship_facts(output_grid)

    input_facts = set(input_relationships["facts"])
    output_facts = set(output_relationships["facts"])

    preserved_relationships = input_facts & output_facts

    behavior_counts = Counter()

    pair_count = min(len(input_objects), len(output_objects))

    for index in range(pair_count):
        behavior = describe_object_behavior(
            input_objects[index],
            output_objects[index],
        )
        behavior_counts[behavior] += 1

    input_h, input_w = grid_shape(input_grid)
    output_h, output_w = grid_shape(output_grid)

    background_preserved = (
        input_candidate["background"] == output_candidate["background"]
    )

    object_count_preserved = len(input_objects) == len(output_objects)

    relationship_count_preserved = len(input_facts) == len(output_facts)

    lines = []

    lines.append("LEARNER THINKING DASHBOARD")
    lines.append("=" * 60)
    lines.append(f"train pair: {pair_index}")
    lines.append("")

    lines.append("1. WHAT I SEE")
    lines.append(f"- input grid size : {input_h}x{input_w}")
    lines.append(f"- output grid size: {output_h}x{output_w}")
    lines.append(f"- input objects   : {len(input_objects)}")
    lines.append(f"- output objects  : {len(output_objects)}")
    lines.append(f"- input relations : {len(input_facts)}")
    lines.append(f"- output relations: {len(output_facts)}")
    lines.append("")

    lines.append("2. WHAT CHANGED")
    lines.append(f"- background preserved: {background_preserved}")
    lines.append(f"- object count preserved: {object_count_preserved}")
    lines.append(f"- relationship count preserved: {relationship_count_preserved}")
    lines.append(
        f"- relationship IDs preserved: "
        f"{len(preserved_relationships)} / {len(input_facts)}"
    )

    if behavior_counts:
        lines.append("- object behavior:")
        for behavior, count in sorted(behavior_counts.items()):
            lines.append(f"  - {behavior}: {count}")
    else:
        lines.append("- object behavior: none detected")

    lines.append("")

    lines.append("3. WHAT I AM TESTING")
    lines.append("- output size may not come from full input grid")
    lines.append("- output size may come from object structure")
    lines.append("- object identity should not trust scan order only")
    lines.append("- containers may shrink")
    lines.append("- small objects may collapse to single cells")
    lines.append("")

    lines.append("4. WHAT I BELIEVE SO FAR")
    if object_count_preserved:
        lines.append("- object count is probably preserved")
    else:
        lines.append("- object count is not proven preserved")

    if relationship_count_preserved:
        lines.append("- relationship count is probably preserved")
    else:
        lines.append("- relationship count is not proven preserved")

    if background_preserved:
        lines.append("- background color is probably preserved")
    else:
        lines.append("- background color is not proven preserved")

    if behavior_counts.get("became_smaller", 0) > 0:
        lines.append("- at least some objects become smaller")

    if behavior_counts.get("became_single_cell", 0) > 0:
        lines.append("- at least some objects become single cells")

    lines.append("5. WHAT I DO NOT KNOW YET")
    lines.append("- exact output size rule")
    lines.append("- exact placement rule")
    lines.append("- exact object identity mapping")
    lines.append("- final test output")

    return "\n".join(lines)


def task_level_dashboard_summary_text_block(train_pairs):
    background_preserved_all = True
    object_count_preserved_all = True
    relationship_count_preserved_all = True
    every_pair_has_smaller = True
    every_pair_has_single_cell = True

    full_input_size_to_output_size = {}

    for pair in train_pairs:
        input_grid = pair["input"]
        output_grid = pair["output"]

        input_h, input_w = grid_shape(input_grid)
        output_h, output_w = grid_shape(output_grid)

        full_input_size_to_output_size.setdefault(
            (input_h, input_w),
            set(),
        ).add((output_h, output_w))

        input_analysis = analyze_grid(input_grid)
        output_analysis = analyze_grid(output_grid)

        input_candidate = select_display_candidate(input_analysis)
        expected_count = input_candidate["component_count"]

        matching_output_candidates = [
            candidate
            for candidate in output_analysis["candidates"]
            if candidate["component_count"] == expected_count
        ]

        if matching_output_candidates:
            output_candidate = matching_output_candidates[0]
        else:
            output_candidate = select_display_candidate(output_analysis)

        input_objects = input_candidate["components"]
        output_objects = output_candidate["components"]

        input_relationships = get_relationship_facts(input_grid)
        output_relationships = get_relationship_facts(output_grid)

        input_facts = set(input_relationships["facts"])
        output_facts = set(output_relationships["facts"])

        if input_candidate["background"] != output_candidate["background"]:
            background_preserved_all = False

        if len(input_objects) != len(output_objects):
            object_count_preserved_all = False

        if len(input_facts) != len(output_facts):
            relationship_count_preserved_all = False

        pair_behavior_counts = Counter()

        pair_count = min(len(input_objects), len(output_objects))

        for index in range(pair_count):
            behavior = describe_object_behavior(
                input_objects[index],
                output_objects[index],
            )
            pair_behavior_counts[behavior] += 1

        if pair_behavior_counts.get("became_smaller", 0) == 0:
            every_pair_has_smaller = False

        if pair_behavior_counts.get("became_single_cell", 0) == 0:
            every_pair_has_single_cell = False

    full_input_grid_rule_failed = any(
        len(output_sizes) > 1
        for output_sizes in full_input_size_to_output_size.values()
    )

    lines = []

    lines.append("TASK VIEW")
    lines.append("=" * 60)
    lines.append("Across all train pairs:")
    lines.append("")

    lines.append("1. STABLE BELIEFS")
    lines.append(
        f"- background preserved all pairs: "
        f"{background_preserved_all}"
    )
    lines.append(
        f"- object count preserved all pairs: "
        f"{object_count_preserved_all}"
    )
    lines.append(
        f"- relationship count preserved all pairs: "
        f"{relationship_count_preserved_all}"
    )
    lines.append(
        f"- every pair has smaller objects: "
        f"{every_pair_has_smaller}"
    )
    lines.append(
        f"- every pair has single-cell objects: "
        f"{every_pair_has_single_cell}"
    )
    lines.append("")

    lines.append("2. SIZE STATUS")
    lines.append(
        f"- full input grid size explains output size: "
        f"{not full_input_grid_rule_failed}"
    )

    if full_input_grid_rule_failed:
        lines.append("- reason: same input size can produce different output sizes")

    lines.append("- output size rule is still unknown")
    lines.append("- next size source to test: object structure")
    lines.append("")

    lines.append("4. NEXT LEARNING TARGET")
    lines.append("- test size sources from objects and relationships")

    return "\n".join(lines)


def make_honest_learned_guess_grid(input_grid, learning_pairs):
    """
    Honest guess maker.

    Important:
    - Does NOT use this input's expected output.
    - Learns only from learning_pairs.
    - If it cannot prove a size rule yet, it returns a visible blank guess
      instead of secretly using the answer.
    """

    input_analysis = analyze_grid(input_grid)
    input_candidate = select_display_candidate(input_analysis)

    input_background = input_candidate["background"]
    input_objects = input_candidate["components"]

    # For now, this is the safest first output-size hypothesis:
    # use the largest visible input object bbox as the guessed canvas.
    #
    # This is NOT claimed correct.
    # The dashboard should later score this hypothesis against train pairs.
    if not input_objects:
        return [[input_background]]

    largest_obj = input_objects[0]
    bbox = largest_obj["bbox"]

    guess_h = max(1, bbox["height"])
    guess_w = max(1, bbox["width"])

    guess_grid = [
        [input_background for _ in range(guess_w)]
        for _ in range(guess_h)
    ]

    foreground_color = largest_obj["color"]

    # Draw a simple normalized version:
    # - container-like objects become bbox outlines
    # - leaf/blob-like objects become single cells
    #
    # This is intentionally simple and visible.
    # It is a first guess, not a forced answer.
    role_facts = get_object_role_facts(input_grid)

    top_offset = bbox["top"]
    left_offset = bbox["left"]

    for role in role_facts:
        obj_bbox = role["bbox"]

        rel_top = obj_bbox["top"] - top_offset
        rel_left = obj_bbox["left"] - left_offset
        rel_bottom = obj_bbox["bottom"] - top_offset
        rel_right = obj_bbox["right"] - left_offset

        if role["contains_count"] > 0:
            for c in range(rel_left, rel_right + 1):
                if 0 <= rel_top < guess_h and 0 <= c < guess_w:
                    guess_grid[rel_top][c] = foreground_color
                if 0 <= rel_bottom < guess_h and 0 <= c < guess_w:
                    guess_grid[rel_bottom][c] = foreground_color

            for r in range(rel_top, rel_bottom + 1):
                if 0 <= r < guess_h and 0 <= rel_left < guess_w:
                    guess_grid[r][rel_left] = foreground_color
                if 0 <= r < guess_h and 0 <= rel_right < guess_w:
                    guess_grid[r][rel_right] = foreground_color

        else:
            center_r = (rel_top + rel_bottom) // 2
            center_c = (rel_left + rel_right) // 2

            if 0 <= center_r < guess_h and 0 <= center_c < guess_w:
                guess_grid[center_r][center_c] = foreground_color

    return guess_grid


def score_guess_against_expected(guess_grid, expected_grid):
    guess_h, guess_w = grid_shape(guess_grid)
    expected_h, expected_w = grid_shape(expected_grid)

    same_size = (guess_h == expected_h and guess_w == expected_w)

    different_cells = 0

    max_h = max(guess_h, expected_h)
    max_w = max(guess_w, expected_w)

    for r in range(max_h):
        for c in range(max_w):
            if r < guess_h and c < guess_w:
                guess_value = guess_grid[r][c]
            else:
                guess_value = None

            if r < expected_h and c < expected_w:
                expected_value = expected_grid[r][c]
            else:
                expected_value = None

            if guess_value != expected_value:
                different_cells += 1

    exact_match = same_size and different_cells == 0

    return {
        "exact_match": exact_match,
        "same_size": same_size,
        "guess_shape": (guess_h, guess_w),
        "expected_shape": (expected_h, expected_w),
        "different_cells": different_cells,
    }


def guess_result_text_block(guess_grid, expected_grid):
    score = score_guess_against_expected(
        guess_grid,
        expected_grid,
    )

    guess_h, guess_w = score["guess_shape"]
    expected_h, expected_w = score["expected_shape"]

    lines = []

    lines.append("GUESS RESULT")
    lines.append("=" * 40)
    lines.append(f"exact match    : {score['exact_match']}")
    lines.append(f"same size      : {score['same_size']}")
    lines.append(f"guess size     : {guess_h}x{guess_w}")
    lines.append(f"expected size  : {expected_h}x{expected_w}")
    lines.append(f"different cells: {score['different_cells']}")

    return "\n".join(lines)

# =============================================================================
# POPUP DRAWING
# =============================================================================

def draw_grid(canvas, grid, x0, y0, cell_size):
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            color = ARC_COLORS.get(value, "#FFFFFF")

            x1 = x0 + c * cell_size
            y1 = y0 + r * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=color,
                outline="#DDDDDD",
            )


def draw_component_overlay(canvas, components, x0, y0, cell_size):
    for index, obj in enumerate(components):
        bbox = obj["bbox"]
        outline = LABEL_COLORS[index % len(LABEL_COLORS)]

        x1 = x0 + bbox["left"] * cell_size
        y1 = y0 + bbox["top"] * cell_size
        x2 = x0 + (bbox["right"] + 1) * cell_size
        y2 = y0 + (bbox["bottom"] + 1) * cell_size

        canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            outline=outline,
            width=3,
        )

        canvas.create_text(
            x1 + 4,
            y1 + 4,
            anchor="nw",
            text=obj["id"],
            fill=outline,
            font=("Arial", 10, "bold"),
        )


def draw_grid_panel(canvas, title, grid, analysis, x0, y0, cell_size):
    canvas.create_text(
        x0,
        y0 - 34,
        anchor="nw",
        text=title,
        fill="black",
        font=("Arial", 15, "bold"),
    )

    display_candidate = select_display_candidate(analysis)

    subtitle = (
        f"colors={list(analysis['color_counts'].keys())} | "
        f"display bg={display_candidate['background']} | "
        f"objects={display_candidate['component_count']}"
    )

    canvas.create_text(
        x0,
        y0 - 15,
        anchor="nw",
        text=subtitle,
        fill="black",
        font=("Arial", 10),
    )

    draw_grid(canvas, grid, x0, y0, cell_size)

    draw_component_overlay(
        canvas,
        display_candidate["components"],
        x0,
        y0,
        cell_size,
    )


def draw_text_card(canvas, title, text, x0, y0, width):
    canvas.create_text(
        x0,
        y0,
        anchor="nw",
        text=title,
        fill="black",
        font=("Arial", 15, "bold"),
    )

    canvas.create_rectangle(
        x0,
        y0 + 28,
        x0 + width,
        y0 + 28 + 260,
        outline="#BBBBBB",
        fill="#FAFAFA",
    )

    canvas.create_text(
        x0 + 12,
        y0 + 40,
        anchor="nw",
        text=text,
        fill="black",
        font=("Consolas", 10),
    )


def size_and_placement_text_block(pair_index, input_grid, output_grid):
    input_h, input_w = grid_shape(input_grid)
    output_h, output_w = grid_shape(output_grid)

    input_analysis = analyze_grid(input_grid)
    output_analysis = analyze_grid(output_grid)
    input_candidate = select_display_candidate(input_analysis)
    expected_count = input_candidate["component_count"]

    matching_output_candidates = [
        candidate
        for candidate in output_analysis["candidates"]
        if candidate["component_count"] == expected_count
    ]

    if matching_output_candidates:
        output_candidate = matching_output_candidates[0]
    else:
        output_candidate = select_display_candidate(output_analysis)

    input_objects = input_candidate["components"]
    output_objects = output_candidate["components"]

    lines = []
    lines.append("LEARNED FACTS - NOT OUTPUT GUESS")
    lines.append("-" * 80)
    lines.append(f"pair: {pair_index}")
    lines.append(f"grid size: {input_h}x{input_w} -> {output_h}x{output_w}")
    lines.append(f"input background : {input_candidate['background']}")
    lines.append(f"output background: {output_candidate['background']}")
    lines.append(f"objects: {len(input_objects)} -> {len(output_objects)}")
    lines.append("")

    pair_count = min(len(input_objects), len(output_objects))

    for index in range(pair_count):
        input_obj = input_objects[index]
        output_obj = output_objects[index]

        input_bbox = input_obj["bbox"]
        output_bbox = output_obj["bbox"]

        behavior = describe_object_behavior(input_obj, output_obj)

        lines.append(
            f"{input_obj['id']} -> {output_obj['id']} | "
            f"color {input_obj['color']}->{output_obj['color']}"
        )
        lines.append(
            f"  bbox "
            f"{input_bbox['height']}x{input_bbox['width']} "
            f"top={input_bbox['top']} left={input_bbox['left']}"
        )
        lines.append(
            f"  -> "
            f"{output_bbox['height']}x{output_bbox['width']} "
            f"top={output_bbox['top']} left={output_bbox['left']}"
        )
        lines.append(f"  behavior: {behavior}")
        lines.append("")

    return "\n".join(lines)


def show_train_pair_popup(pair_index, input_grid, output_grid):
    input_analysis = analyze_grid(input_grid)
    output_analysis = analyze_grid(output_grid)

    learning_pairs = [
        pair
        for index, pair in enumerate(CURRENT_TRAIN_PAIRS_FOR_POPUP)
        if index != pair_index
    ]

    guess_grid = make_honest_learned_guess_grid(
        input_grid,
        learning_pairs,
    )

    guess_analysis = analyze_grid(guess_grid)

    guess_result_text = guess_result_text_block(
        guess_grid,
        output_grid,
    )

    dashboard_text = learner_dashboard_text_block(
        pair_index,
        input_grid,
        output_grid,
        CURRENT_TRAIN_PAIRS_FOR_POPUP,
    )

    cell_size = 20
    gap = 60
    left_margin = 30
    top_margin = 90

    input_h, input_w = grid_shape(input_grid)
    output_h, output_w = grid_shape(output_grid)

    guess_h, guess_w = grid_shape(guess_grid)

    input_px_w = input_w * cell_size
    input_px_h = input_h * cell_size

    output_px_w = output_w * cell_size
    output_px_h = output_h * cell_size

    guess_px_w = guess_w * cell_size
    guess_px_h = guess_h * cell_size

    input_x = left_margin
    output_x = input_x + input_px_w + gap
    guess_x = output_x + output_px_w + gap

    grid_bottom_y = top_margin + max(
        input_px_h,
        output_px_h,
        guess_px_h,
    )

    dashboard_x = left_margin
    dashboard_y = grid_bottom_y + 80
    dashboard_width = 1000
    guess_result_x = guess_x
    guess_result_y = grid_bottom_y + 80
    guess_result_width = 360

    dashboard_y = guess_result_y + 220

    dashboard_line_count = len(dashboard_text.splitlines())
    dashboard_height = 80 + (dashboard_line_count * 18)

    canvas_width = max(
        guess_x + guess_px_w + left_margin,
        dashboard_x + dashboard_width + left_margin,
    )
    canvas_height = dashboard_y + dashboard_height + 80


    root = tk.Tk()
    root.title(f"TRAIN PAIR {pair_index} | VISUAL STORY VIEW")

    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True)

    visible_width = min(canvas_width, 1600)
    visible_height = min(canvas_height, 950)

    canvas = tk.Canvas(
        frame,
        width=visible_width,
        height=visible_height,
        scrollregion=(0, 0, canvas_width, canvas_height),
        bg="white",
    )

    y_scroll = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    x_scroll = tk.Scrollbar(frame, orient="horizontal", command=canvas.xview)

    canvas.configure(
        yscrollcommand=y_scroll.set,
        xscrollcommand=x_scroll.set,
    )

    canvas.grid(row=0, column=0, sticky="nsew")
    y_scroll.grid(row=0, column=1, sticky="ns")
    x_scroll.grid(row=1, column=0, sticky="ew")

    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)

    draw_grid_panel(
        canvas,
        "INPUT",
        input_grid,
        input_analysis,
        input_x,
        top_margin,
        cell_size,
    )

    draw_grid_panel(
        canvas,
        "EXPECTED",
        output_grid,
        output_analysis,
        output_x,
        top_margin,
        cell_size,
    )

    draw_grid_panel(
        canvas,
        "GUESS",
        guess_grid,
        guess_analysis,
        guess_x,
        top_margin,
        cell_size,
    )
    draw_text_card(
        canvas,
        "GUESS RESULT",
        guess_result_text,
        guess_result_x,
        guess_result_y,
        guess_result_width,
    )
    draw_text_card(
        canvas,
        "LEARNER THINKING DASHBOARD",
        dashboard_text,
        dashboard_x,
        dashboard_y,
        dashboard_width,
    )

    root.mainloop()


def show_test_pair_popup(test_index, input_grid, train_pairs):
    input_analysis = analyze_grid(input_grid)


    cell_size = 20
    gap = 60
    left_margin = 30
    top_margin = 90

    input_h, input_w = grid_shape(input_grid)

    input_px_w = input_w * cell_size

    input_px_h = input_h * cell_size

    max_grid_h = input_px_h

    input_x = left_margin
    canvas_width = input_x + input_px_w + left_margin
    canvas_height = top_margin + max_grid_h + 120

    root = tk.Tk()
    root.title(f"TEST PAIR {test_index} | VISUAL STORY VIEW")

    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(
        frame,
        width=min(canvas_width, 1600),
        height=min(canvas_height, 950),
        scrollregion=(0, 0, canvas_width, canvas_height),
        bg="white",
    )

    y_scroll = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    x_scroll = tk.Scrollbar(frame, orient="horizontal", command=canvas.xview)

    canvas.configure(
        yscrollcommand=y_scroll.set,
        xscrollcommand=x_scroll.set,
    )

    canvas.grid(row=0, column=0, sticky="nsew")
    y_scroll.grid(row=0, column=1, sticky="ns")
    x_scroll.grid(row=1, column=0, sticky="ew")

    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)

    draw_grid_panel(
        canvas,
        "TEST INPUT RAW + COMPONENTS",
        input_grid,
        input_analysis,
        input_x,
        top_margin,
        cell_size,
    )


    root.mainloop()


def show_all_popups(task):
    global CURRENT_TRAIN_PAIRS_FOR_POPUP

    CURRENT_TRAIN_PAIRS_FOR_POPUP = task.get("train", [])
    for pair_index, pair in enumerate(task.get("train", [])):
        show_train_pair_popup(
            pair_index=pair_index,
            input_grid=pair["input"],
            output_grid=pair["output"],
        )

    for test_index, pair in enumerate(task.get("test", [])):
        show_test_pair_popup(
            test_index=test_index,
            input_grid=pair["input"],
            train_pairs=task["train"],
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    if len(sys.argv) > 1:
        task_path = sys.argv[1]
    else:
        task_path = DEFAULT_TASK_PATH

    if not os.path.exists(task_path):
        raise FileNotFoundError(task_path)

    task, wrapped_key = load_task(task_path)

    print()
    print("RAW VISUAL COMPONENT DEBUGGER")
    print("=" * 80)
    print(f"Task path: {task_path}")

    if wrapped_key is not None:
        print(f"Using wrapped task key: {wrapped_key}")

    print(f"Train pairs: {len(task.get('train', []))}")
    print(f"Test pairs : {len(task.get('test', []))}")

    print_task_component_summary(task)
    show_all_popups(task)


if __name__ == "__main__":
    main()