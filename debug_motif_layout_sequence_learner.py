# debug_motif_layout_sequence_learner.py
import json
import os
from collections import deque


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


def crop(grid, top, left, bottom, right):
    return [
        row[left:right + 1]
        for row in grid[top:bottom + 1]
    ]


def nonzero_colors(grid):
    colors = set()

    for row in grid:
        for value in row:
            if value != BACKGROUND:
                colors.add(value)

    return sorted(colors)


# ============================================================
# COMPONENTS
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

            if value == BACKGROUND:
                continue

            if value in ignore_colors:
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

            shape = []
            for row in patch:
                shape.append(tuple(1 if cell != BACKGROUND else 0 for cell in row))

            components.append({
                "color": color,
                "top": top,
                "left": left,
                "bottom": bottom,
                "right": right,
                "height": bottom - top + 1,
                "width": right - left + 1,
                "cell_count": len(cells),
                "patch": patch,
                "shape": tuple(shape),
            })

    components.sort(key=lambda item: (item["top"], item["left"], item["color"]))
    return components


# ============================================================
# SOURCE BLOCKS
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

        components = extract_components(raw)

        blocks.append({
            "index": len(blocks),
            "top": start,
            "bottom": end,
            "height": end - start + 1,
            "width": w,
            "colors": nonzero_colors(raw),
            "raw": raw,
            "components": components,
        })

    return blocks


def component_token(component):
    return {
        "color": component["color"],
        "height": component["height"],
        "width": component["width"],
        "cell_count": component["cell_count"],
        "shape": component["shape"],
    }


def block_slot_components(block):
    comps = list(block["components"])
    comps.sort(key=lambda item: (item["left"], item["top"], item["color"]))
    return comps


def source_sequence_row_major(blocks):
    seq = []

    for block in blocks:
        for comp in block_slot_components(block):
            token = component_token(comp)
            token["block_index"] = block["index"]
            token["slot_index"] = len(seq)
            seq.append(token)

    return seq


def source_sequence_column_major(blocks):
    slots_by_block = [
        block_slot_components(block)
        for block in blocks
    ]

    max_slots = max((len(slots) for slots in slots_by_block), default=0)

    seq = []

    for slot_index in range(max_slots):
        for block_index, slots in enumerate(slots_by_block):
            if slot_index >= len(slots):
                continue

            comp = slots[slot_index]
            token = component_token(comp)
            token["block_index"] = block_index
            token["slot_index"] = slot_index
            seq.append(token)

    return seq


# ============================================================
# EXPECTED EVENT DECOMPOSITION
# ============================================================

def horizontal_runs_for_component(component):
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
                "cell_count": end - start + 1,
                "kind": "horizontal_run",
            })

    return events


def vertical_runs_for_component(component, split_height_2=False):
    events = []
    patch = component["patch"]
    height = len(patch)
    width = len(patch[0]) if height else 0

    for local_c in range(width):
        r = 0

        while r < height:
            while r < height and patch[r][local_c] == BACKGROUND:
                r += 1

            if r >= height:
                break

            start = r

            while r < height and patch[r][local_c] != BACKGROUND:
                r += 1

            end = r - 1
            run_height = end - start + 1

            if split_height_2 and run_height > 2:
                part_top = start

                while part_top <= end:
                    part_height = min(2, end - part_top + 1)

                    events.append({
                        "color": component["color"],
                        "top": component["top"] + part_top,
                        "left": component["left"] + local_c,
                        "height": part_height,
                        "width": 1,
                        "cell_count": part_height,
                        "kind": "vertical_run_split2",
                    })

                    part_top += part_height
            else:
                events.append({
                    "color": component["color"],
                    "top": component["top"] + start,
                    "left": component["left"] + local_c,
                    "height": run_height,
                    "width": 1,
                    "cell_count": run_height,
                    "kind": "vertical_run",
                })

    return events


def expected_primitive_events(expected_grid, anchor, split_six_height_2=False):
    components = extract_components(
        expected_grid,
        ignore_colors={ANCHOR_COLOR},
    )

    events = []

    for component in components:
        color = component["color"]

        if color in (1, 2, 3):
            events.extend(horizontal_runs_for_component(component))
        elif color == 6:
            events.extend(
                vertical_runs_for_component(
                    component,
                    split_height_2=split_six_height_2,
                )
            )
        else:
            events.append({
                "color": color,
                "top": component["top"],
                "left": component["left"],
                "height": component["height"],
                "width": component["width"],
                "cell_count": component["cell_count"],
                "kind": "component",
            })

    ar, ac = anchor if anchor is not None else (0, 0)

    for event in events:
        event["rel_top"] = event["top"] - ar
        event["rel_left"] = event["left"] - ac

    events.sort(key=lambda item: (item["top"], item["left"], item["color"]))

    return events


# ============================================================
# SEQUENCE SCORING
# ============================================================

def colors_of(tokens):
    return [item["color"] for item in tokens]


def sequence_score(source_colors, expected_colors):
    score = 0

    for a, b in zip(source_colors, expected_colors):
        if a == b:
            score += 1

    exact = source_colors == expected_colors

    return {
        "score": score,
        "exact": exact,
        "source_len": len(source_colors),
        "expected_len": len(expected_colors),
    }


def print_token_sequence(title, tokens):
    print()
    print(title)
    print("  colors:", colors_of(tokens))

    for idx, token in enumerate(tokens, start=1):
        print(
            f"  {idx}. color={token['color']} "
            f"size={token['height']}x{token['width']} "
            f"cells={token['cell_count']} "
            f"block={token.get('block_index')} "
            f"slot={token.get('slot_index')}"
        )


def print_event_sequence(title, events):
    print()
    print(title)
    print("  colors:", colors_of(events))

    for idx, event in enumerate(events, start=1):
        print(
            f"  {idx}. color={event['color']} "
            f"kind={event['kind']} "
            f"size={event['height']}x{event['width']} "
            f"rel=({event['rel_top']},{event['rel_left']}) "
            f"cells={event['cell_count']}"
        )


# ============================================================
# ANALYZE PAIR
# ============================================================

def analyze_pair(pair_index, pair):
    input_grid = pair["input"]
    expected_grid = pair["output"]

    divider_col = find_divider_col(input_grid)

    if divider_col is None:
        return None

    left_grid, right_grid = split_by_divider(input_grid, divider_col)
    anchor = find_anchor(right_grid)

    blocks = extract_left_blocks(left_grid)

    row_major = source_sequence_row_major(blocks)
    column_major = source_sequence_column_major(blocks)

    expected_no_split = expected_primitive_events(
        expected_grid,
        anchor,
        split_six_height_2=False,
    )

    expected_split6 = expected_primitive_events(
        expected_grid,
        anchor,
        split_six_height_2=True,
    )

    return {
        "pair_index": pair_index,
        "divider_col": divider_col,
        "anchor": anchor,
        "block_count": len(blocks),
        "blocks": blocks,
        "row_major": row_major,
        "column_major": column_major,
        "expected_no_split": expected_no_split,
        "expected_split6": expected_split6,
    }


def print_pair_analysis(info):
    print()
    print("=" * 80)
    print(f"TRAIN PAIR {info['pair_index'] + 1}")
    print("=" * 80)
    print(f"divider_col: {info['divider_col']}")
    print(f"anchor     : {info['anchor']}")
    print(f"blocks     : {info['block_count']}")

    print()
    print("SOURCE BLOCKS")
    for block in info["blocks"]:
        print(
            f"  block {block['index'] + 1}: "
            f"rows={block['top']}..{block['bottom']} "
            f"colors={block['colors']}"
        )

        for comp_idx, comp in enumerate(block_slot_components(block), start=1):
            print(
                f"    slot {comp_idx}: color={comp['color']} "
                f"size={comp['height']}x{comp['width']} "
                f"cells={comp['cell_count']}"
            )

    print_token_sequence("SOURCE ROW-MAJOR", info["row_major"])
    print_token_sequence("SOURCE COLUMN-MAJOR", info["column_major"])

    print_event_sequence(
        "EXPECTED PRIMITIVES - 6 NOT SPLIT",
        info["expected_no_split"],
    )

    print_event_sequence(
        "EXPECTED PRIMITIVES - 6 SPLIT INTO HEIGHT-2",
        info["expected_split6"],
    )

    for expected_name in ["expected_no_split", "expected_split6"]:
        expected_colors = colors_of(info[expected_name])

        row_score = sequence_score(colors_of(info["row_major"]), expected_colors)
        col_score = sequence_score(colors_of(info["column_major"]), expected_colors)

        print()
        print(f"SEQUENCE MATCH AGAINST {expected_name}")
        print(f"  row_major   : {row_score}")
        print(f"  column_major: {col_score}")


# ============================================================
# LEARN BEST SOURCE ORDER
# ============================================================

def learn_best_source_order(pair_infos):
    candidates = [
        ("row_major", "expected_no_split"),
        ("row_major", "expected_split6"),
        ("column_major", "expected_no_split"),
        ("column_major", "expected_split6"),
    ]

    scored = []

    for source_name, expected_name in candidates:
        total_score = 0
        exact_count = 0
        length_penalty = 0

        for info in pair_infos:
            source_colors = colors_of(info[source_name])
            expected_colors = colors_of(info[expected_name])
            result = sequence_score(source_colors, expected_colors)

            total_score += result["score"]

            if result["exact"]:
                exact_count += 1

            length_penalty += abs(
                result["source_len"] - result["expected_len"]
            )

        scored.append({
            "source_name": source_name,
            "expected_name": expected_name,
            "total_score": total_score,
            "exact_count": exact_count,
            "length_penalty": length_penalty,
        })

    scored.sort(
        key=lambda item: (
            item["exact_count"],
            item["total_score"],
            -item["length_penalty"],
        ),
        reverse=True,
    )

    return scored


def print_learned_order_summary(pair_infos, test_info):
    print()
    print("=" * 80)
    print("LEARNED SOURCE-ORDER SUMMARY")
    print("=" * 80)

    scored = learn_best_source_order(pair_infos)

    for idx, item in enumerate(scored, start=1):
        print(
            f"{idx}. source={item['source_name']} "
            f"expected={item['expected_name']} "
            f"exact={item['exact_count']}/{len(pair_infos)} "
            f"score={item['total_score']} "
            f"length_penalty={item['length_penalty']}"
        )

    best = scored[0]

    print()
    print("BEST SOURCE ORDER")
    print(best)

    if test_info is not None:
        print()
        print("=" * 80)
        print("TEST SOURCE SEQUENCE USING BEST ORDER")
        print("=" * 80)
        print(f"test anchor: {test_info['anchor']}")
        print(f"test blocks: {test_info['block_count']}")

        tokens = test_info[best["source_name"]]
        print_token_sequence("TEST PREDICTED EVENT COLOR SEQUENCE", tokens)


# ============================================================
# MAIN
# ============================================================

def main():
    task_id_or_path = input("Task id or json path: ").strip()
    task_id, task = load_task(task_id_or_path)

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    print()
    print("=" * 80)
    print(f"MOTIF LAYOUT SEQUENCE LEARNER: {task_id}")
    print("=" * 80)

    pair_infos = []

    for pair_index, pair in enumerate(train_pairs):
        info = analyze_pair(pair_index, pair)

        if info is None:
            print(f"TRAIN PAIR {pair_index + 1}: could not analyze")
            continue

        pair_infos.append(info)
        print_pair_analysis(info)

    test_info = None

    if test_pairs:
        fake_pair = {
            "input": test_pairs[0]["input"],
            "output": [[0]],
        }

        # Analyze test source side only.
        divider_col = find_divider_col(fake_pair["input"])

        if divider_col is not None:
            left_grid, right_grid = split_by_divider(fake_pair["input"], divider_col)
            anchor = find_anchor(right_grid)
            blocks = extract_left_blocks(left_grid)

            test_info = {
                "anchor": anchor,
                "block_count": len(blocks),
                "blocks": blocks,
                "row_major": source_sequence_row_major(blocks),
                "column_major": source_sequence_column_major(blocks),
            }

    print_learned_order_summary(pair_infos, test_info)


if __name__ == "__main__":
    main()