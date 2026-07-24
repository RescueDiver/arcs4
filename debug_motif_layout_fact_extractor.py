# debug_motif_layout_fact_extractor.py
import json
import os
from collections import deque, defaultdict

from core.grid_utils import print_grid
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
# GRID HELPERS
# ============================================================

def grid_shape(grid):
    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def find_divider_col(grid, divider_color=DIVIDER_COLOR):
    h, w = grid_shape(grid)

    best_col = None
    best_count = -1

    for c in range(w):
        count = 0
        for r in range(h):
            if grid[r][c] == divider_color:
                count += 1

        if count > best_count:
            best_count = count
            best_col = c

    if best_count == h:
        return best_col

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
# LEFT BLOCK EXTRACTION
# ============================================================

def row_has_nonzero(row):
    return any(value != BACKGROUND for value in row)


def extract_left_blocks(left_grid):
    """
    Extract row blocks separated by blank rows.
    This avoids hardcoding 3-row chunks only.
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
            "colors": nonzero_colors(raw),
            "raw": raw,
        })

    return blocks


# ============================================================
# COMPONENT EXTRACTION
# ============================================================

def component_cells(grid, start_r, start_c, visited, include_anchor=True):
    h, w = grid_shape(grid)
    color = grid[start_r][start_c]

    q = deque()
    q.append((start_r, start_c))
    visited.add((start_r, start_c))

    cells = []

    while q:
        r, c = q.popleft()
        cells.append((r, c))

        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc

            if rr < 0 or cc < 0 or rr >= h or cc >= w:
                continue

            if (rr, cc) in visited:
                continue

            if grid[rr][cc] != color:
                continue

            visited.add((rr, cc))
            q.append((rr, cc))

    return color, cells


def normalize_component(grid, cells):
    top = min(r for r, _ in cells)
    bottom = max(r for r, _ in cells)
    left = min(c for _, c in cells)
    right = max(c for _, c in cells)

    patch = crop(grid, top, left, bottom, right)

    signature = []
    for row in patch:
        signature.append(tuple(1 if value != BACKGROUND else 0 for value in row))

    return {
        "top": top,
        "left": left,
        "bottom": bottom,
        "right": right,
        "height": bottom - top + 1,
        "width": right - left + 1,
        "cell_count": len(cells),
        "shape_signature": tuple(signature),
        "patch": patch,
    }


def extract_components(grid, include_anchor=True):
    h, w = grid_shape(grid)
    visited = set()
    components = []

    for r in range(h):
        for c in range(w):
            value = grid[r][c]

            if value == BACKGROUND:
                continue

            if not include_anchor and value == ANCHOR_COLOR:
                continue

            if (r, c) in visited:
                continue

            color, cells = component_cells(
                grid,
                r,
                c,
                visited,
                include_anchor=include_anchor,
            )

            if not include_anchor and color == ANCHOR_COLOR:
                continue

            info = normalize_component(grid, cells)
            info["color"] = color
            info["cells"] = cells
            components.append(info)

    components.sort(key=lambda item: (item["top"], item["left"], item["color"]))
    return components


# ============================================================
# PRINT HELPERS
# ============================================================

def describe_component(component, anchor=None):
    if anchor is None:
        rel = ""
    else:
        ar, ac = anchor
        rel = (
            f" rel_to_anchor=({component['top'] - ar}, "
            f"{component['left'] - ac})"
        )

    return (
        f"color={component['color']} "
        f"bbox=({component['top']},{component['left']})-"
        f"({component['bottom']},{component['right']}) "
        f"size={component['height']}x{component['width']} "
        f"cells={component['cell_count']}"
        f"{rel}"
    )


def print_components(title, components, anchor=None):
    print()
    print(title)

    if not components:
        print("  None")
        return

    for idx, component in enumerate(components, start=1):
        print(f"  {idx}. {describe_component(component, anchor=anchor)}")


def print_block_summary(block):
    print()
    print(f"LEFT BLOCK {block['index'] + 1}")
    print(f"  rows  : {block['top']}..{block['bottom']}")
    print(f"  shape : {block['height']}x{block['width']}")
    print(f"  colors: {block['colors']}")
    print_grid(block["raw"], "  RAW BLOCK")

    components = extract_components(block["raw"], include_anchor=False)
    print_components("  COMPONENTS", components)


# ============================================================
# PAIR ANALYSIS
# ============================================================

def analyze_pair(task_id, pair_index, pair):
    input_grid = pair["input"]
    expected_grid = pair["output"]

    divider_col = find_divider_col(input_grid)

    print()
    print("=" * 80)
    print(f"TRAIN PAIR {pair_index + 1}")
    print("=" * 80)

    print(f"input shape   : {grid_shape(input_grid)}")
    print(f"expected shape: {grid_shape(expected_grid)}")
    print(f"divider col   : {divider_col}")

    if divider_col is None:
        print("No full divider column found.")
        return None

    left_grid, right_grid = split_by_divider(input_grid, divider_col)
    anchor = find_anchor(right_grid)

    print(f"left shape    : {grid_shape(left_grid)}")
    print(f"right shape   : {grid_shape(right_grid)}")
    print(f"anchor in right grid: {anchor}")

    print_grid(input_grid, "INPUT")
    print_grid(expected_grid, "EXPECTED")

    show_three_grids(
        input_grid,
        expected_grid,
        expected_grid,
        title_a=f"{task_id} TRAIN {pair_index + 1} INPUT",
        title_b="EXPECTED",
        title_c="EXPECTED",
    )

    blocks = extract_left_blocks(left_grid)

    print()
    print(f"LEFT BLOCK COUNT: {len(blocks)}")

    for block in blocks:
        print_block_summary(block)

    expected_components = extract_components(
        expected_grid,
        include_anchor=False,
    )

    print_components(
        "EXPECTED OUTPUT COMPONENTS",
        expected_components,
        anchor=anchor,
    )

    expected_event_sequence = []

    for component in expected_components:
        if component["color"] == ANCHOR_COLOR:
            continue

        ar, ac = anchor if anchor is not None else (0, 0)

        expected_event_sequence.append({
            "color": component["color"],
            "height": component["height"],
            "width": component["width"],
            "cell_count": component["cell_count"],
            "top": component["top"],
            "left": component["left"],
            "rel_top": component["top"] - ar,
            "rel_left": component["left"] - ac,
            "shape_signature": component["shape_signature"],
        })

    print()
    print("EXPECTED EVENT SEQUENCE")
    for idx, event in enumerate(expected_event_sequence, start=1):
        print(
            f"  {idx}. color={event['color']} "
            f"size={event['height']}x{event['width']} "
            f"rel=({event['rel_top']},{event['rel_left']}) "
            f"cells={event['cell_count']}"
        )

    return {
        "pair_index": pair_index,
        "divider_col": divider_col,
        "anchor": anchor,
        "block_count": len(blocks),
        "blocks": blocks,
        "expected_events": expected_event_sequence,
    }


def summarize_across_pairs(pair_infos):
    print()
    print("=" * 80)
    print("ACROSS-PAIR SUMMARY")
    print("=" * 80)

    by_block_count = defaultdict(list)

    for info in pair_infos:
        if info is None:
            continue

        by_block_count[info["block_count"]].append(info)

    print()
    print("BLOCK COUNTS SEEN")
    for block_count, infos in sorted(by_block_count.items()):
        pair_nums = [info["pair_index"] + 1 for info in infos]
        print(f"  block_count={block_count}: train pairs {pair_nums}")

    print()
    print("EXPECTED EVENTS BY TRAIN PAIR")
    for info in pair_infos:
        if info is None:
            continue

        print()
        print(f"TRAIN PAIR {info['pair_index'] + 1}")
        print(f"  anchor={info['anchor']}")
        print(f"  block_count={info['block_count']}")

        for idx, event in enumerate(info["expected_events"], start=1):
            print(
                f"    {idx}. color={event['color']} "
                f"size={event['height']}x{event['width']} "
                f"rel=({event['rel_top']},{event['rel_left']})"
            )


# ============================================================
# MAIN
# ============================================================

def main():
    task_id_or_path = input("Task id or json path: ").strip()
    task_id, task = load_task(task_id_or_path)

    train_pairs = task.get("train", [])

    print()
    print("=" * 80)
    print(f"MOTIF LAYOUT FACT EXTRACTOR: {task_id}")
    print("=" * 80)

    pair_infos = []

    for pair_index, pair in enumerate(train_pairs):
        info = analyze_pair(task_id, pair_index, pair)
        pair_infos.append(info)

    summarize_across_pairs(pair_infos)


if __name__ == "__main__":
    main()