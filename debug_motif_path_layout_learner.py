# debug_motif_path_layout_learner.py
import json
import os
from collections import deque, defaultdict

from OLD.arc_visualizer import show_three_grids


DIVIDER_COLOR = 4
ANCHOR_COLOR = 5
BACKGROUND = 0


# ============================================================
# LOAD TASK
# ============================================================

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


# ============================================================
# BASIC HELPERS
# ============================================================

def grid_shape(grid):
    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def print_grid(title, grid):
    h, w = grid_shape(grid)
    print(f"{title} (h={h}, w={w})")
    for row in grid:
        print(" ".join(str(v) for v in row))


def find_divider_col(grid):
    h, w = grid_shape(grid)

    for c in range(w):
        if all(grid[r][c] == DIVIDER_COLOR for r in range(h)):
            return c

    return None


def split_by_divider(grid, divider_col):
    left = [row[:divider_col] for row in grid]
    right = [row[divider_col + 1:] for row in grid]
    return left, right


def find_anchor(grid):
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == ANCHOR_COLOR:
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


def colors_of(items):
    return [item["color"] for item in items]


def score_prediction(predicted, expected):
    if predicted is None:
        return 0, False

    if len(predicted) != len(expected):
        return 0, False

    if expected and len(predicted[0]) != len(expected[0]):
        return 0, False

    score = 0

    for r in range(len(expected)):
        for c in range(len(expected[0])):
            if predicted[r][c] == expected[r][c]:
                score += 1

    return score, predicted == expected


# ============================================================
# LEFT BLOCK / SLOT EXTRACTION
# ============================================================

def extract_left_blocks(left_grid):
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
    seen = set()
    collapsed = []

    for slot in block["slots"]:
        color = slot["color"]

        if color in seen:
            continue

        seen.add(color)
        collapsed.append(slot)

    return collapsed


# ============================================================
# SOURCE SEQUENCE MODES
# ============================================================

def source_row_major(blocks, collapse_duplicates=False):
    seq = []

    for block in blocks:
        slots = (
            collapse_duplicate_colors_in_block(block)
            if collapse_duplicates
            else block["slots"]
        )

        for slot in slots:
            seq.append(dict(slot))

    return seq


def source_column_major(blocks, collapse_duplicates=False):
    block_slots = []

    for block in blocks:
        slots = (
            collapse_duplicate_colors_in_block(block)
            if collapse_duplicates
            else block["slots"]
        )
        block_slots.append(slots)

    max_slots = max((len(slots) for slots in block_slots), default=0)

    seq = []

    for slot_index in range(max_slots):
        for slots in block_slots:
            if slot_index >= len(slots):
                continue

            seq.append(dict(slots[slot_index]))

    return seq


def build_source_sequences(blocks):
    return {
        "row_major_all_slots": source_row_major(blocks, collapse_duplicates=False),
        "column_major_all_slots": source_column_major(blocks, collapse_duplicates=False),
        "row_major_unique_colors": source_row_major(blocks, collapse_duplicates=True),
        "column_major_unique_colors": source_column_major(blocks, collapse_duplicates=True),
    }


# ============================================================
# EXPECTED PRIMITIVE EVENTS
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

            color = value
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

                    if grid[nr][nc] != color:
                        continue

                    visited.add((nr, nc))
                    q.append((nr, nc))

            top = min(x for x, _ in cells)
            bottom = max(x for x, _ in cells)
            left = min(y for _, y in cells)
            right = max(y for _, y in cells)

            patch = crop(grid, top, left, bottom, right)

            components.append({
                "color": color,
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

    all_exact_color_sequences = True

    for info in train_infos:
        source_seq = info["source_sequences"][sequence_mode]
        expected_events = info["expected_events"]

        if colors_of(source_seq) != colors_of(expected_events):
            all_exact_color_sequences = False
            break

        endpoint = info["expected_anchor"]

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

    if not all_exact_color_sequences:
        return None

    learned_sizes = {}
    learned_directions = {}

    for color, sizes in size_by_color.items():
        if len(sizes) != 1:
            return None
        learned_sizes[color] = sorted(sizes)[0]

    for color, directions in direction_by_color.items():
        if len(directions) != 1:
            return None
        learned_directions[color] = sorted(directions)[0]

    return {
        "sequence_mode": sequence_mode,
        "learned_sizes": learned_sizes,
        "learned_directions": learned_directions,
    }


def place_path_events(output_h, output_w, anchor, source_seq, learned_rule):
    output = [
        [BACKGROUND for _ in range(output_w)]
        for _ in range(output_h)
    ]

    if anchor is None:
        return None

    ar, ac = anchor

    if not (0 <= ar < output_h and 0 <= ac < output_w):
        return None

    output[ar][ac] = ANCHOR_COLOR
    endpoint = anchor

    for token in source_seq:
        color = token["color"]

        if color not in learned_rule["learned_sizes"]:
            return None

        if color not in learned_rule["learned_directions"]:
            return None

        height, width = learned_rule["learned_sizes"][color]
        direction = learned_rule["learned_directions"][color]

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

def analyze_train_pair(pair_index, pair):
    input_grid = pair["input"]
    expected_grid = pair["output"]

    divider_col = find_divider_col(input_grid)

    if divider_col is None:
        return None

    left_grid, right_grid = split_by_divider(input_grid, divider_col)
    right_anchor = find_anchor(right_grid)
    expected_anchor = find_anchor(expected_grid)

    blocks = add_slots_to_blocks(
        extract_left_blocks(left_grid)
    )

    source_sequences = build_source_sequences(blocks)
    expected_events = expected_primitive_events(expected_grid)

    return {
        "pair_index": pair_index,
        "input_grid": input_grid,
        "expected_grid": expected_grid,
        "divider_col": divider_col,
        "right_anchor": right_anchor,
        "expected_anchor": expected_anchor,
        "output_shape": grid_shape(expected_grid),
        "blocks": blocks,
        "source_sequences": source_sequences,
        "expected_events": expected_events,
    }


def analyze_test_pair(pair):
    input_grid = pair["input"]

    divider_col = find_divider_col(input_grid)

    if divider_col is None:
        return None

    left_grid, right_grid = split_by_divider(input_grid, divider_col)
    anchor = find_anchor(right_grid)

    blocks = add_slots_to_blocks(
        extract_left_blocks(left_grid)
    )

    source_sequences = build_source_sequences(blocks)

    h, right_w = grid_shape(right_grid)

    return {
        "input_grid": input_grid,
        "divider_col": divider_col,
        "anchor": anchor,
        "output_shape": (h, right_w),
        "blocks": blocks,
        "source_sequences": source_sequences,
    }


# ============================================================
# LEARN / VERIFY
# ============================================================

def learn_candidate_rules(train_infos):
    sequence_modes = [
        "column_major_all_slots",
        "column_major_unique_colors",
        "row_major_all_slots",
        "row_major_unique_colors",
    ]

    candidates = []

    for mode in sequence_modes:
        learned_rule = learn_path_properties_for_mode(
            train_infos,
            mode,
        )

        if learned_rule is None:
            candidates.append({
                "mode": mode,
                "learned": False,
                "train_exact": 0,
                "train_score": 0,
                "rule": None,
            })
            continue

        train_exact = 0
        train_score = 0

        for info in train_infos:
            output_h, output_w = info["output_shape"]

            predicted = place_path_events(
                output_h,
                output_w,
                info["expected_anchor"],
                info["source_sequences"][mode],
                learned_rule,
            )

            score, exact = score_prediction(
                predicted,
                info["expected_grid"],
            )

            train_score += score

            if exact:
                train_exact += 1

        candidates.append({
            "mode": mode,
            "learned": True,
            "train_exact": train_exact,
            "train_score": train_score,
            "rule": learned_rule,
        })

    candidates.sort(
        key=lambda item: (
            item["train_exact"],
            item["train_score"],
            item["learned"],
        ),
        reverse=True,
    )

    return candidates


def print_train_info(info):
    print()
    print("=" * 80)
    print(f"TRAIN PAIR {info['pair_index'] + 1}")
    print("=" * 80)
    print(f"divider_col    : {info['divider_col']}")
    print(f"right anchor   : {info['right_anchor']}")
    print(f"expected anchor: {info['expected_anchor']}")
    print(f"output shape   : {info['output_shape']}")
    print()

    for name, seq in info["source_sequences"].items():
        print(f"{name}: {colors_of(seq)}")

    print(f"expected events: {colors_of(info['expected_events'])}")


def print_rule(rule):
    print(f"sequence_mode: {rule['sequence_mode']}")
    print(f"learned_sizes: {rule['learned_sizes']}")
    print(f"directions   : {rule['learned_directions']}")


# ============================================================
# MAIN
# ============================================================

def main():
    task_id_or_path = input("Task id or json path: ").strip()
    task_id, task = load_task(task_id_or_path)

    print()
    print("=" * 80)
    print(f"MOTIF PATH LAYOUT LEARNER: {task_id}")
    print("=" * 80)

    train_infos = []

    for pair_index, pair in enumerate(task.get("train", [])):
        info = analyze_train_pair(pair_index, pair)

        if info is None:
            print(f"TRAIN PAIR {pair_index + 1}: could not analyze")
            continue

        train_infos.append(info)
        print_train_info(info)

    candidates = learn_candidate_rules(train_infos)

    print()
    print("=" * 80)
    print("LEARNED PATH CANDIDATES")
    print("=" * 80)

    for idx, candidate in enumerate(candidates, start=1):
        print()
        print(f"{idx}. mode={candidate['mode']}")
        print(f"   learned    : {candidate['learned']}")
        print(f"   train exact: {candidate['train_exact']}/{len(train_infos)}")
        print(f"   train score: {candidate['train_score']}")

        if candidate["rule"] is not None:
            print_rule(candidate["rule"])

    exact_candidates = [
        candidate for candidate in candidates
        if candidate["learned"]
        and candidate["train_exact"] == len(train_infos)
    ]

    if not exact_candidates:
        print()
        print("No exact learned path candidate.")
        return

    print()
    print("=" * 80)
    print("VISUAL TRAIN CHECKS")
    print("=" * 80)

    best = exact_candidates[0]

    for info in train_infos:
        output_h, output_w = info["output_shape"]

        predicted = place_path_events(
            output_h,
            output_w,
            info["expected_anchor"],
            info["source_sequences"][best["mode"]],
            best["rule"],
        )

        score, exact = score_prediction(
            predicted,
            info["expected_grid"],
        )

        print()
        print(f"TRAIN PAIR {info['pair_index'] + 1}")
        print(f"mode : {best['mode']}")
        print(f"score: {score}")
        print(f"exact: {exact}")
        print_grid("EXPECTED", info["expected_grid"])
        print_grid("PREDICTED", predicted)

        show_three_grids(
            info["input_grid"],
            info["expected_grid"],
            predicted,
            title_a=f"{task_id} TRAIN {info['pair_index'] + 1} INPUT",
            title_b="EXPECTED",
            title_c=f"PREDICTED {best['mode']}",
        )

    print()
    print("=" * 80)
    print("TEST PREDICTIONS")
    print("=" * 80)

    test_pairs = task.get("test", [])

    for test_index, pair in enumerate(test_pairs):
        test_info = analyze_test_pair(pair)

        if test_info is None:
            print(f"TEST PAIR {test_index + 1}: could not analyze")
            continue

        print()
        print("-" * 80)
        print(f"TEST PAIR {test_index + 1}")
        print(f"anchor      : {test_info['anchor']}")
        print(f"output shape: {test_info['output_shape']}")

        # Show the best exact candidate first.
        shown = 0

        for candidate in exact_candidates[:2]:
            output_h, output_w = test_info["output_shape"]

            predicted = place_path_events(
                output_h,
                output_w,
                test_info["anchor"],
                test_info["source_sequences"][candidate["mode"]],
                candidate["rule"],
            )

            print()
            print(f"TEST GUESS {shown + 1}")
            print(f"mode: {candidate['mode']}")

            if candidate["rule"] is not None:
                print_rule(candidate["rule"])

            if predicted is not None:
                print_grid("TEST PREDICTED", predicted)

                show_three_grids(
                    test_info["input_grid"],
                    predicted,
                    predicted,
                    title_a=f"{task_id} TEST {test_index + 1} INPUT",
                    title_b=f"TEST GUESS {shown + 1}",
                    title_c=f"TEST GUESS {shown + 1}",
                )
            else:
                print("TEST PREDICTED: None")

            shown += 1


if __name__ == "__main__":
    main()