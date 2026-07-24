# visual_motif_layout_review.py
import json
import os
import tkinter as tk
from collections import deque


DIVIDER_COLOR = 4
ANCHOR_COLOR = 5
BACKGROUND = 0

ARC_COLORS = {
    0: "#111111",
    1: "#0074D9",
    2: "#FF4136",
    3: "#2ECC40",
    4: "#FFDC00",
    5: "#AAAAAA",
    6: "#F012BE",
    7: "#FF851B",
    8: "#7FDBFF",
    9: "#870C25",
}


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
# HELPERS
# ============================================================

def grid_shape(grid):
    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


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


# ============================================================
# SOURCE BLOCK / SLOT EXTRACTION
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
            "top": block["top"],
            "left": left,
            "height": block["height"],
            "width": right - left + 1,
        })

    return slots


def source_overlays_for_input(input_grid):
    divider_col = find_divider_col(input_grid)

    overlays = []

    if divider_col is None:
        return overlays

    h, _ = grid_shape(input_grid)

    overlays.append({
        "top": 0,
        "left": divider_col,
        "height": h,
        "width": 1,
        "label": "DIV",
        "outline": "yellow",
    })

    left_grid, right_grid = split_by_divider(input_grid, divider_col)
    blocks = extract_left_blocks(left_grid)

    for block in blocks:
        overlays.append({
            "top": block["top"],
            "left": 0,
            "height": block["height"],
            "width": divider_col,
            "label": f"B{block['index'] + 1}",
            "outline": "white",
        })

        slots = extract_slots_from_block(block)

        for slot in slots:
            overlays.append({
                "top": slot["top"],
                "left": slot["left"],
                "height": slot["height"],
                "width": slot["width"],
                "label": f"B{slot['block_index'] + 1}S{slot['slot_index'] + 1}:{slot['color']}",
                "outline": "cyan",
            })

    anchor = find_anchor(right_grid)

    if anchor is not None:
        ar, ac = anchor
        overlays.append({
            "top": ar,
            "left": divider_col + 1 + ac,
            "height": 1,
            "width": 1,
            "label": "A",
            "outline": "red",
        })

    return overlays


# ============================================================
# EXPECTED EVENT EXTRACTION
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

            components.append({
                "color": color,
                "top": top,
                "left": left,
                "bottom": bottom,
                "right": right,
                "height": bottom - top + 1,
                "width": right - left + 1,
            })

    components.sort(key=lambda item: (item["top"], item["left"], item["color"]))
    return components


def expected_event_overlays(expected_grid):
    overlays = []

    components = extract_components(
        expected_grid,
        ignore_colors={ANCHOR_COLOR},
    )

    event_index = 1

    for comp in components:
        overlays.append({
            "top": comp["top"],
            "left": comp["left"],
            "height": comp["height"],
            "width": comp["width"],
            "label": f"E{event_index}:{comp['color']}",
            "outline": "lime",
        })
        event_index += 1

    anchor = find_anchor(expected_grid)

    if anchor is not None:
        ar, ac = anchor
        overlays.append({
            "top": ar,
            "left": ac,
            "height": 1,
            "width": 1,
            "label": "A",
            "outline": "red",
        })

    return overlays


# ============================================================
# DRAWING
# ============================================================

def readable_text_color(value):
    if value in (0, 9):
        return "white"
    return "black"


def draw_grid(parent, grid, title, overlays=None, cell=26):
    if overlays is None:
        overlays = []

    h, w = grid_shape(grid)

    frame = tk.Frame(parent, bg="#222222")
    frame.pack(side=tk.LEFT, padx=12, pady=8, anchor="n")

    label = tk.Label(
        frame,
        text=f"{title}  ({h}x{w})",
        fg="white",
        bg="#222222",
        font=("Consolas", 12, "bold"),
    )
    label.pack(anchor="w")

    canvas = tk.Canvas(
        frame,
        width=w * cell + 2,
        height=h * cell + 2,
        bg="#222222",
        highlightthickness=0,
    )
    canvas.pack()

    for r in range(h):
        for c in range(w):
            value = grid[r][c]
            color = ARC_COLORS.get(value, "#999999")

            x1 = c * cell
            y1 = r * cell
            x2 = x1 + cell
            y2 = y1 + cell

            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=color,
                outline="#333333",
            )

            if value != 0:
                canvas.create_text(
                    x1 + cell / 2,
                    y1 + cell / 2,
                    text=str(value),
                    fill=readable_text_color(value),
                    font=("Consolas", 9, "bold"),
                )

    for overlay in overlays:
        top = overlay["top"]
        left = overlay["left"]
        height = overlay["height"]
        width = overlay["width"]
        label_text = overlay["label"]
        outline = overlay.get("outline", "white")

        x1 = left * cell
        y1 = top * cell
        x2 = (left + width) * cell
        y2 = (top + height) * cell

        canvas.create_rectangle(
            x1 + 2,
            y1 + 2,
            x2 - 2,
            y2 - 2,
            outline=outline,
            width=3,
        )

        canvas.create_rectangle(
            x1 + 2,
            y1 + 2,
            x1 + 8 + len(label_text) * 7,
            y1 + 18,
            fill="black",
            outline=outline,
        )

        canvas.create_text(
            x1 + 5,
            y1 + 10,
            text=label_text,
            fill=outline,
            font=("Consolas", 8, "bold"),
            anchor="w",
        )

    return frame


def make_scroll_window(title):
    root = tk.Tk()
    root.title(title)

    outer = tk.Frame(root)
    outer.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(outer, bg="#111111")
    scrollbar_y = tk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar_x = tk.Scrollbar(outer, orient=tk.HORIZONTAL, command=canvas.xview)

    content = tk.Frame(canvas, bg="#111111")

    content.bind(
        "<Configure>",
        lambda event: canvas.configure(
            scrollregion=canvas.bbox("all")
        ),
    )

    canvas.create_window((0, 0), window=content, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

    root.geometry("1400x900")

    return root, content


def add_pair_row(parent, title, input_grid, expected_grid=None):
    section = tk.Frame(parent, bg="#111111")
    section.pack(fill=tk.X, padx=8, pady=16, anchor="w")

    header = tk.Label(
        section,
        text=title,
        fg="white",
        bg="#111111",
        font=("Consolas", 16, "bold"),
    )
    header.pack(anchor="w")

    row = tk.Frame(section, bg="#111111")
    row.pack(anchor="w")

    draw_grid(
        row,
        input_grid,
        "INPUT: source blocks + slots",
        overlays=source_overlays_for_input(input_grid),
    )

    if expected_grid is not None:
        draw_grid(
            row,
            expected_grid,
            "EXPECTED: output events",
            overlays=expected_event_overlays(expected_grid),
        )


# ============================================================
# MAIN
# ============================================================

def main():
    task_id_or_path = input("Task id or json path: ").strip()
    task_id, task = load_task(task_id_or_path)

    root, content = make_scroll_window(
        f"Motif Layout Visual Review - {task_id}"
    )

    title = tk.Label(
        content,
        text=f"MOTIF LAYOUT VISUAL REVIEW: {task_id}",
        fg="white",
        bg="#111111",
        font=("Consolas", 18, "bold"),
    )
    title.pack(anchor="w", padx=8, pady=8)

    for pair_index, pair in enumerate(task.get("train", [])):
        add_pair_row(
            content,
            f"TRAIN PAIR {pair_index + 1}",
            pair["input"],
            pair["output"],
        )

    for test_index, pair in enumerate(task.get("test", [])):
        add_pair_row(
            content,
            f"TEST PAIR {test_index + 1}",
            pair["input"],
            None,
        )

    root.mainloop()


if __name__ == "__main__":
    main()
