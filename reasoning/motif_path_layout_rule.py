# reasoning/motif_path_layout_rule.py
"""
Learned motif path layout rule.

Purpose:
- Replace the old fixed motif_layout_rule.py schedules.
- Learn from train pairs:
    1. divider color
    2. anchor color
    3. source slot read order
    4. output size per color
    5. output direction per color
- Apply learned path to test input.

This is a task-level learner, not a fixed placement schedule.
"""

import json
import os
import sys
from collections import deque, defaultdict


DIVIDER_COLOR = 4
ANCHOR_COLOR = 5
BACKGROUND = 0


# ============================================================
# BASIC GRID HELPERS
# ============================================================

def grid_shape(grid):
    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def print_grid(title, grid):
    if grid is None:
        print(f"{title}: None")
        return

    h, w = grid_shape(grid)
    print(f"{title} (h={h}, w={w})")

    for row in grid:
        print(" ".join(str(v) for v in row))


def grids_equal(a, b):
    return a == b


def score_prediction(predicted, expected):
    if predicted is None:
        return 0, False

    if predicted == expected:
        h, w = grid_shape(expected)
        return 1000000 + (h * w), True

    ph, pw = grid_shape(predicted)
    eh, ew = grid_shape(expected)

    if ph != eh or pw != ew:
        return 0, False

    score = 0

    for r in range(eh):
        for c in range(ew):
            if predicted[r][c] == expected[r][c]:
                score += 1

    return score, False


def colors_of(items):
    return [item["color"] for item in items]


# ============================================================
# INPUT SPLITTING
# ============================================================

def find_divider_col(grid, divider_color=DIVIDER_COLOR):
    h, w = grid_shape(grid)

    for c in range(w):
        if all(grid[r][c] == divider_color for r in range(h)):
            return c

    return None


def split_by_divider(grid, divider_col):
    left = [row[:divider_col] for row in grid]
    right = [row[divider_col + 1:] for row in grid]
    return left, right


def find_anchor(grid, anchor_color=ANCHOR_COLOR):
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == anchor_color:
                return r, c

    return None


def row_has_nonzero(row):
    return any(value != BACKGROUND for value in row)


def col_has_nonzero(grid, c):
    return any(row[c] != BACKGROUND for row in grid)


def crop(grid, top, left, bottom, right):
    return [
        row[left:right + 1]
        for row in grid[top:bottom + 1]
    ]


def dominant_color(grid):
    counts = {}

    for row in grid:
        for value in row:
            if value == BACKGROUND:
                continue

            counts[value] = counts.get(value, 0) + 1

    if not counts:
        return None

    return max(counts, key=counts.get)


# ============================================================
# SOURCE BLOCK / SLOT EXTRACTION
# ============================================================

def extract_left_blocks(left_grid):
    """
    Left side is split into row-blocks separated by blank rows.
    """
    blocks = []
    h, w = grid_shape(left_grid)

    r = 0

    while r < h:
        while r < h and not row_has_nonzero(left_grid[r]):
            r += 1

        if r >= h:
            break

        start = r

        while r < h and row_has_nonzero(left_grid[r]):
            r += 1

        end = r - 1
        raw = crop(left_grid, start, 0, end, w - 1)

        blocks.append({
            "index": len(blocks),
            "top": start,
            "bottom": end,
            "height": end - start + 1,
            "width": w,
            "raw": raw,
        })

    return blocks


def extract_column_groups(grid):
    """
    Inside each source block, slots are vertical column groups.
    This matched the visual review for 136b0064.
    """
    h, w = grid_shape(grid)
    groups = []

    c = 0

    while c < w:
        while c < w and not col_has_nonzero(grid, c):
            c += 1

        if c >= w:
            break

        start = c

        while c < w and col_has_nonzero(grid, c):
            c += 1

        end = c - 1
        groups.append((start, end))

    return groups


def extract_slots_from_block(block):
    raw = block["raw"]
    groups = extract_column_groups(raw)

    slots = []

    for slot_index, (left, right) in enumerate(groups):
        patch = crop(raw, 0, left, len(raw) - 1, right)
        color = dominant_color(patch)

        if color is None:
            continue

        slots.append({
            "block_index": block["index"],
            "slot_index": slot_index,
            "color": color,
            "height": len(patch),
            "width": len(patch[0]) if patch else 0,
            "patch": patch,
        })

    return slots


def add_slots_to_blocks(blocks):
    for block in blocks:
        block["slots"] = extract_slots_from_block(block)

    return blocks


def collapse_duplicate_colors_in_block(block):
    """
    For guess diversity:
    - all_slots keeps every slot.
    - unique_colors keeps first occurrence of each color inside a block.
    """
    seen = set()
    collapsed = []

    for slot in block["slots"]:
        color = slot["color"]

        if color in seen:
            continue

        seen.add(color)
        collapsed.append(slot)

    return collapsed


def source_column_major(blocks, collapse_duplicates=False):
    block_slots = []

    for block in blocks:
        if collapse_duplicates:
            slots = collapse_duplicate_colors_in_block(block)
        else:
            slots = block["slots"]

        block_slots.append(slots)

    max_slots = max((len(slots) for slots in block_slots), default=0)
    sequence = []

    for slot_index in range(max_slots):
        for slots in block_slots:
            if slot_index >= len(slots):
                continue

            sequence.append(dict(slots[slot_index]))

    return sequence


def source_row_major(blocks, collapse_duplicates=False):
    sequence = []

    for block in blocks:
        if collapse_duplicates:
            slots = collapse_duplicate_colors_in_block(block)
        else:
            slots = block["slots"]

        for slot in slots:
            sequence.append(dict(slot))

    return sequence


def build_source_sequences(blocks):
    return {
        "column_major_unique_colors": source_column_major(
            blocks,
            collapse_duplicates=True,
        ),
        "column_major_all_slots": source_column_major(
            blocks,
            collapse_duplicates=False,
        ),
        "row_major_unique_colors": source_row_major(
            blocks,
            collapse_duplicates=True,
        ),
        "row_major_all_slots": source_row_major(
            blocks,
            collapse_duplicates=False,
        ),
    }


# ============================================================
# EXPECTED OUTPUT EVENT EXTRACTION
# ============================================================

def extract_components(grid, ignore_colors=None):
    if ignore_colors is None:
        ignore_colors = set()

    h, w = grid_shape(grid)
    visited = set()
    components = []

    for r in range(h):
        for c in range(w):
            value = grid[r][c]

            if value == BACKGROUND or value in ignore_colors:
                continue

            if (r, c) in visited:
                continue

            q = deque([(r, c)])
            visited.add((r, c))
            cells = []

            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))

                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc

                    if nr < 0 or nc < 0 or nr >= h or nc >= w:
                        continue

                    if (nr, nc) in visited:
                        continue

                    if grid[nr][nc] != value:
                        continue

                    visited.add((nr, nc))
                    q.append((nr, nc))

            top = min(x for x, _ in cells)
            bottom = max(x for x, _ in cells)
            left = min(y for _, y in cells)
            right = max(y for _, y in cells)

            patch = crop(grid, top, left, bottom, right)

            components.append({
                "color": value,
                "top": top,
                "left": left,
                "bottom": bottom,
                "right": right,
                "height": bottom - top + 1,
                "width": right - left + 1,
                "patch": patch,
            })

    components.sort(key=lambda item: (item["top"], item["left"], item["color"]))
    return components


def horizontal_run_events(component):
    events = []
    patch = component["patch"]

    for local_r, row in enumerate(patch):
        c = 0

        while c < len(row):
            while c < len(row) and row[c] == BACKGROUND:
                c += 1

            if c >= len(row):
                break

            start = c

            while c < len(row) and row[c] != BACKGROUND:
                c += 1

            end = c - 1

            events.append({
                "color": component["color"],
                "top": component["top"] + local_r,
                "left": component["left"] + start,
                "height": 1,
                "width": end - start + 1,
            })

    return events


def vertical_run_events(component, split_height=2):
    events = []
    patch = component["patch"]
    h = len(patch)
    w = len(patch[0]) if h else 0

    for local_c in range(w):
        r = 0

        while r < h:
            while r < h and patch[r][local_c] == BACKGROUND:
                r += 1

            if r >= h:
                break

            start = r

            while r < h and patch[r][local_c] != BACKGROUND:
                r += 1

            end = r - 1
            part = start

            while part <= end:
                part_height = min(split_height, end - part + 1)

                events.append({
                    "color": component["color"],
                    "top": component["top"] + part,
                    "left": component["left"] + local_c,
                    "height": part_height,
                    "width": 1,
                })

                part += part_height

    return events


def expected_primitive_events(expected_grid):
    components = extract_components(
        expected_grid,
        ignore_colors={ANCHOR_COLOR},
    )

    events = []

    for component in components:
        color = component["color"]

        if color in (1, 2, 3):
            events.extend(horizontal_run_events(component))
        elif color == 6:
            events.extend(vertical_run_events(component, split_height=2))
        else:
            events.append({
                "color": color,
                "top": component["top"],
                "left": component["left"],
                "height": component["height"],
                "width": component["width"],
            })

    events.sort(key=lambda item: (item["top"], item["left"], item["color"]))
    return events


# ============================================================
# PATH LEARNING
# ============================================================

def event_direction_from_endpoint(event, endpoint):
    endpoint_r, endpoint_c = endpoint

    if event["height"] > 1 and event["width"] == 1:
        if event["left"] == endpoint_c and event["top"] == endpoint_r + 1:
            return "down"

        return None

    if event["height"] == 1 and event["width"] > 1:
        if event["top"] != endpoint_r + 1:
            return None

        if event["left"] == endpoint_c:
            return "right"

        if event["left"] + event["width"] - 1 == endpoint_c:
            return "left"

        return None

    return None


def endpoint_after_event(event, direction):
    if direction == "down":
        return event["top"] + event["height"] - 1, event["left"]

    if direction == "right":
        return event["top"], event["left"] + event["width"] - 1

    if direction == "left":
        return event["top"], event["left"]

    return None


def learn_path_properties_for_mode(train_infos, sequence_mode):
    size_by_color = defaultdict(set)
    direction_by_color = defaultdict(set)

    for info in train_infos:
        source_seq = info["source_sequences"][sequence_mode]
        expected_events = info["expected_events"]

        if colors_of(source_seq) != colors_of(expected_events):
            return None

        endpoint = info["expected_anchor"]

        if endpoint is None:
            return None

        for source_token, event in zip(source_seq, expected_events):
            color = source_token["color"]

            size_by_color[color].add((event["height"], event["width"]))

            direction = event_direction_from_endpoint(event, endpoint)

            if direction is None:
                return None

            direction_by_color[color].add(direction)

            endpoint = endpoint_after_event(event, direction)

            if endpoint is None:
                return None

    learned_sizes = {}
    learned_directions = {}

    for color, sizes in size_by_color.items():
        if len(sizes) != 1:
            return None

        learned_sizes[color] = next(iter(sizes))

    for color, directions in direction_by_color.items():
        if len(directions) != 1:
            return None

        learned_directions[color] = next(iter(directions))

    return {
        "sequence_mode": sequence_mode,
        "learned_sizes": learned_sizes,
        "learned_directions": learned_directions,
    }


def place_path_events(output_h, output_w, anchor, source_seq, learned_rule):
    if anchor is None:
        return None

    output = [
        [BACKGROUND for _ in range(output_w)]
        for _ in range(output_h)
    ]

    ar, ac = anchor

    if not (0 <= ar < output_h and 0 <= ac < output_w):
        return None

    output[ar][ac] = ANCHOR_COLOR
    endpoint = anchor

    sizes = learned_rule["learned_sizes"]
    directions = learned_rule["learned_directions"]

    for token in source_seq:
        color = token["color"]

        if color not in sizes:
            return None

        if color not in directions:
            return None

        height, width = sizes[color]
        direction = directions[color]
        endpoint_r, endpoint_c = endpoint

        if direction == "down":
            top = endpoint_r + 1
            left = endpoint_c

        elif direction == "right":
            top = endpoint_r + 1
            left = endpoint_c

        elif direction == "left":
            top = endpoint_r + 1
            left = endpoint_c - width + 1

        else:
            return None

        if top < 0 or left < 0:
            return None

        if top + height > output_h:
            return None

        if left + width > output_w:
            return None

        for r in range(top, top + height):
            for c in range(left, left + width):
                output[r][c] = color

        event = {
            "color": color,
            "top": top,
            "left": left,
            "height": height,
            "width": width,
        }

        endpoint = endpoint_after_event(event, direction)

        if endpoint is None:
            return None

    return output


# ============================================================
# PAIR ANALYSIS
# ============================================================

def analyze_input(input_grid):
    divider_col = find_divider_col(input_grid)

    if divider_col is None:
        return None

    left_grid, right_grid = split_by_divider(input_grid, divider_col)
    anchor = find_anchor(right_grid)
    blocks = add_slots_to_blocks(extract_left_blocks(left_grid))
    source_sequences = build_source_sequences(blocks)

    return {
        "divider_col": divider_col,
        "left_grid": left_grid,
        "right_grid": right_grid,
        "anchor": anchor,
        "output_shape": grid_shape(right_grid),
        "blocks": blocks,
        "source_sequences": source_sequences,
    }


def analyze_train_pair(pair_index, pair):
    input_info = analyze_input(pair["input"])

    if input_info is None:
        return None

    expected_grid = pair["output"]
    expected_anchor = find_anchor(expected_grid)
    expected_events = expected_primitive_events(expected_grid)

    info = dict(input_info)
    info.update({
        "pair_index": pair_index,
        "input_grid": pair["input"],
        "expected_grid": expected_grid,
        "expected_anchor": expected_anchor,
        "expected_events": expected_events,
        "expected_shape": grid_shape(expected_grid),
    })

    return info


# ============================================================
# PUBLIC LEARN / APPLY API
# ============================================================

def learn_motif_path_layout_rule(train_pairs):
    train_infos = []

    for pair_index, pair in enumerate(train_pairs):
        info = analyze_train_pair(pair_index, pair)

        if info is None:
            return None

        train_infos.append(info)

    if not train_infos:
        return None

    preferred_modes = [
        "column_major_unique_colors",
        "column_major_all_slots",
        "row_major_unique_colors",
        "row_major_all_slots",
    ]

    candidate_rules = []

    for mode in preferred_modes:
        learned_rule = learn_path_properties_for_mode(train_infos, mode)

        if learned_rule is None:
            continue

        exact_count = 0
        total_score = 0
        train_results = []

        for info in train_infos:
            output_h, output_w = info["output_shape"]

            predicted = place_path_events(
                output_h,
                output_w,
                info["anchor"],
                info["source_sequences"][mode],
                learned_rule,
            )

            score, exact = score_prediction(
                predicted,
                info["expected_grid"],
            )

            if exact:
                exact_count += 1

            total_score += score

            train_results.append({
                "pair_index": info["pair_index"],
                "score": score,
                "exact": exact,
            })

        candidate_rules.append({
            "mode": mode,
            "learned_sizes": learned_rule["learned_sizes"],
            "learned_directions": learned_rule["learned_directions"],
            "exact_count": exact_count,
            "total_score": total_score,
            "train_results": train_results,
        })

    exact_candidates = [
        rule for rule in candidate_rules
        if rule["exact_count"] == len(train_infos)
    ]

    if not exact_candidates:
        return None

    exact_candidates.sort(
        key=lambda rule: (
            rule["exact_count"],
            rule["total_score"],
            -preferred_modes.index(rule["mode"]),
        ),
        reverse=True,
    )

    return {
        "family": "motif_path_layout_rule",
        "rule_type": "learned_column_major_path_layout",
        "train_pair_count": len(train_infos),
        "train_exact_count": len(train_infos),
        "candidate_rules": exact_candidates,
        "divider_color": DIVIDER_COLOR,
        "anchor_color": ANCHOR_COLOR,
        "background": BACKGROUND,
    }


def apply_motif_path_layout_rule(rule, input_grid, guess_index=0):
    if rule is None:
        return None

    candidates = rule.get("candidate_rules", [])

    if not candidates:
        return None

    if guess_index < 0 or guess_index >= len(candidates):
        return None

    input_info = analyze_input(input_grid)

    if input_info is None:
        return None

    candidate = candidates[guess_index]
    mode = candidate["mode"]

    source_seq = input_info["source_sequences"].get(mode)

    if source_seq is None:
        return None

    output_h, output_w = input_info["output_shape"]

    learned_rule = {
        "sequence_mode": mode,
        "learned_sizes": candidate["learned_sizes"],
        "learned_directions": candidate["learned_directions"],
    }

    return place_path_events(
        output_h,
        output_w,
        input_info["anchor"],
        source_seq,
        learned_rule,
    )


def generate_motif_path_layout_guesses(rule, input_grid, max_guesses=2):
    guesses = []

    for guess_index in range(max_guesses):
        predicted = apply_motif_path_layout_rule(
            rule,
            input_grid,
            guess_index=guess_index,
        )

        if predicted is None:
            continue

        if predicted in guesses:
            continue

        guesses.append(predicted)

    return guesses


def solve_motif_path_layout(train_pairs, test_input_grid):
    rule = learn_motif_path_layout_rule(train_pairs)

    if rule is None:
        return []

    return generate_motif_path_layout_guesses(
        rule,
        test_input_grid,
        max_guesses=2,
    )


# ============================================================
# OPTIONAL SELF-TEST RUNNER
# ============================================================

def load_json_file(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def unwrap_task(raw, task_id):
    if "train" in raw:
        return raw

    if task_id in raw:
        return raw[task_id]

    if isinstance(raw, dict) and len(raw) == 1:
        return raw[next(iter(raw))]

    raise KeyError(task_id)


def load_task_for_self_test(task_id_or_path):
    project_root = os.path.dirname(os.path.dirname(__file__))
    value = task_id_or_path.strip().strip('"')

    if value.endswith(".json") and os.path.exists(value):
        raw = load_json_file(value)
        task_id = os.path.splitext(os.path.basename(value))[0]
        return task_id, unwrap_task(raw, task_id)

    failure_path = os.path.join(
        project_root,
        "data_failures",
        "extracted_tasks",
        value + ".json",
    )

    if os.path.exists(failure_path):
        raw = load_json_file(failure_path)
        return value, unwrap_task(raw, value)

    data_path = os.path.join(project_root, "data", "data.json")

    if os.path.exists(data_path):
        data = load_json_file(data_path)

        if value in data:
            return value, data[value]

    raise FileNotFoundError(value)


def self_test(task_id_or_path):
    task_id, task = load_task_for_self_test(task_id_or_path)
    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    print()
    print("=" * 80)
    print(f"MOTIF PATH LAYOUT RULE SELF-TEST: {task_id}")
    print("=" * 80)

    rule = learn_motif_path_layout_rule(train_pairs)

    if rule is None:
        print("No learned rule.")
        return

    print()
    print("LEARNED RULE")
    print(f"family           : {rule['family']}")
    print(f"rule_type        : {rule['rule_type']}")
    print(f"train exact count: {rule['train_exact_count']}/{rule['train_pair_count']}")

    for idx, candidate in enumerate(rule["candidate_rules"], start=1):
        print()
        print(f"CANDIDATE {idx}")
        print(f"mode      : {candidate['mode']}")
        print(f"sizes     : {candidate['learned_sizes']}")
        print(f"directions: {candidate['learned_directions']}")
        print(f"exact     : {candidate['exact_count']}/{rule['train_pair_count']}")
        print(f"score     : {candidate['total_score']}")

    print()
    print("=" * 80)
    print("TRAIN CHECK")
    print("=" * 80)

    train_exact = 0

    for pair_index, pair in enumerate(train_pairs):
        predicted = apply_motif_path_layout_rule(
            rule,
            pair["input"],
            guess_index=0,
        )

        score, exact = score_prediction(predicted, pair["output"])

        if exact:
            train_exact += 1

        print()
        print(f"TRAIN PAIR {pair_index + 1}")
        print(f"score: {score}")
        print(f"exact: {exact}")

        print_grid("EXPECTED", pair["output"])
        print_grid("PREDICTED", predicted)

    print()
    print(f"TRAIN EXACT: {train_exact}/{len(train_pairs)}")

    print()
    print("=" * 80)
    print("TEST GUESSES")
    print("=" * 80)

    for test_index, pair in enumerate(test_pairs):
        guesses = generate_motif_path_layout_guesses(
            rule,
            pair["input"],
            max_guesses=2,
        )

        for guess_index, guess in enumerate(guesses, start=1):
            print()
            print(f"TEST PAIR {test_index + 1} GUESS {guess_index}")
            print_grid("TEST PREDICTED", guess)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        self_test(sys.argv[1])
    else:
        task_value = input("Task id or json path: ").strip()
        self_test(task_value)