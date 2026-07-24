import json
import os
import tkinter as tk


CELL = 26

COLORS = {
    0: "#000000",
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
# BASIC HELPERS
# ============================================================

def grid_shape(grid):
    if grid is None:
        return 0, 0

    h = len(grid)
    w = len(grid[0]) if h else 0

    return h, w


def color_counts(grid):
    counts = {}

    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1

    return counts


def most_common_color(grid):
    counts = color_counts(grid)

    if not counts:
        return 0

    return max(counts, key=counts.get)


def colors_in_grid(grid):
    colors = set()

    for row in grid:
        for value in row:
            colors.add(value)

    return colors


def bounding_box(cells):
    if not cells:
        return None

    rows = [r for r, c in cells]
    cols = [c for r, c in cells]

    return {
        "top": min(rows),
        "left": min(cols),
        "bottom": max(rows),
        "right": max(cols),
        "height": max(rows) - min(rows) + 1,
        "width": max(cols) - min(cols) + 1,
    }


def bbox_text(box):
    if box is None:
        return "None"

    return (
        f"top={box['top']} left={box['left']} "
        f"h={box['height']} w={box['width']}"
    )


def in_bounds(grid, r, c):
    h, w = grid_shape(grid)
    return 0 <= r < h and 0 <= c < w


# ============================================================
# OBJECT DETECTION
# ============================================================

def find_components_by_color_set(grid, allowed_colors, background_color=None):
    """
    Connected components where cells may be any color in allowed_colors.

    This is important for anchor tasks because a yellow+orange object
    should be seen as one object.
    """
    h, w = grid_shape(grid)

    seen = set()
    components = []

    for r in range(h):
        for c in range(w):
            value = grid[r][c]

            if value not in allowed_colors:
                continue

            if background_color is not None and value == background_color:
                continue

            if (r, c) in seen:
                continue

            stack = [(r, c)]
            seen.add((r, c))
            cells = []

            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = rr + dr
                    nc = cc + dc

                    if not in_bounds(grid, nr, nc):
                        continue

                    if (nr, nc) in seen:
                        continue

                    if grid[nr][nc] not in allowed_colors:
                        continue

                    seen.add((nr, nc))
                    stack.append((nr, nc))

            counts = {}

            for rr, cc in cells:
                value = grid[rr][cc]
                counts[value] = counts.get(value, 0) + 1

            components.append({
                "cells": cells,
                "bbox": bounding_box(cells),
                "cell_count": len(cells),
                "colors": sorted(counts.keys()),
                "color_counts": counts,
            })

    components.sort(
        key=lambda obj: (
            obj["bbox"]["top"],
            obj["bbox"]["left"],
            obj["cell_count"],
        )
    )

    return components


def component_has_color(component, color):
    return color in component["color_counts"]


def component_anchor_cells(grid, component, anchor_color):
    cells = []

    for r, c in component["cells"]:
        if grid[r][c] == anchor_color:
            cells.append((r, c))

    return cells


def component_summary_lines(grid, components, anchor_color=None):
    lines = []

    for i, comp in enumerate(components, start=1):
        line = (
            f"object {i}: "
            f"colors={comp['colors']} "
            f"cells={comp['cell_count']} "
            f"bbox=({bbox_text(comp['bbox'])})"
        )

        if anchor_color is not None and component_has_color(comp, anchor_color):
            anchors = component_anchor_cells(grid, comp, anchor_color)
            line += f" anchor_cells={anchors}"

        lines.append(line)

    if not lines:
        lines.append("No objects found.")

    return lines


# ============================================================
# SIMPLE STORY ANALYSIS
# ============================================================

def analyze_pair(input_grid, output_grid):
    input_bg = most_common_color(input_grid)

    input_colors = colors_in_grid(input_grid)
    output_colors = colors_in_grid(output_grid)

    # Important:
    # In cropped outputs, the object color may be more common than the background.
    # So prefer the input background if it still exists in the output.
    if input_bg in output_colors:
        output_bg = input_bg
    else:
        output_bg = most_common_color(output_grid)

    disappeared = sorted(input_colors - output_colors)
    appeared = sorted(output_colors - input_colors)
    shared = sorted(input_colors & output_colors)

    anchor_color = disappeared[0] if disappeared else None

    output_structure_colors = sorted(output_colors - {output_bg})
    structure_color = output_structure_colors[0] if output_structure_colors else None

    input_allowed = set(input_colors) - {input_bg}
    output_allowed = set(output_colors) - {output_bg}

    input_components = find_components_by_color_set(
        input_grid,
        input_allowed,
        background_color=input_bg,
    )

    output_components = find_components_by_color_set(
        output_grid,
        output_allowed,
        background_color=output_bg,
    )

    anchor_components = []

    if anchor_color is not None:
        for comp in input_components:
            if component_has_color(comp, anchor_color):
                anchor_components.append(comp)

    story = []

    story.append(f"input shape : {grid_shape(input_grid)}")
    story.append(f"output shape: {grid_shape(output_grid)}")
    story.append("")
    story.append(f"input background : {input_bg}")
    story.append(f"output background: {output_bg}")
    story.append(f"input colors : {sorted(input_colors)}")
    story.append(f"output colors: {sorted(output_colors)}")
    story.append("")
    story.append(f"shared colors     : {shared}")
    story.append(f"disappeared colors: {disappeared}")
    story.append(f"appeared colors   : {appeared}")
    story.append("")
    story.append(f"guessed anchor color   : {anchor_color}")
    story.append(f"guessed structure color: {structure_color}")
    story.append("")
    story.append(f"input object count : {len(input_components)}")
    story.append(f"output object count: {len(output_components)}")
    story.append(f"anchor-object count: {len(anchor_components)}")
    story.append("")

    if anchor_color is not None:
        if len(anchor_components) == 2:
            story.append("HYPOTHESIS: two input objects contain the anchor color.")
            story.append("This supports anchor-relative merge / overlay.")
        elif len(anchor_components) == 1:
            story.append("HYPOTHESIS: only one input object contains the anchor color.")
            story.append("Overlay may need split-by-anchor, not component-by-component.")
        elif len(anchor_components) == 0:
            story.append("HYPOTHESIS: anchor color disappeared, but no anchor object was found.")
        else:
            story.append("HYPOTHESIS: many anchor objects; may be a repeated marker task.")

    return {
        "input_bg": input_bg,
        "output_bg": output_bg,
        "anchor_color": anchor_color,
        "structure_color": structure_color,
        "input_components": input_components,
        "output_components": output_components,
        "anchor_components": anchor_components,
        "story_lines": story,
    }


# ============================================================
# TASK LOADING
# ============================================================

def load_task(task_id):
    base_dir = os.path.dirname(__file__)

    data_path = os.path.join(base_dir, "data", "data.json")
    failure_path = os.path.join(
        base_dir,
        "data_failures",
        "extracted_tasks",
        task_id + ".json",
    )

    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if task_id in data:
            return task_id, data[task_id]

    if os.path.exists(failure_path):
        with open(failure_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if "train" in raw:
            return task_id, raw

        if task_id in raw:
            return task_id, raw[task_id]

        if isinstance(raw, dict) and len(raw) == 1:
            key = next(iter(raw))
            return key, raw[key]

    raise FileNotFoundError(task_id)


# ============================================================
# DRAWING
# ============================================================

def draw_grid(parent, title, grid):
    box = tk.Frame(parent)
    box.pack(side=tk.LEFT, padx=10, pady=8, anchor="n")

    h, w = grid_shape(grid)

    tk.Label(
        box,
        text=f"{title}  {h}x{w}",
        font=("Arial", 11, "bold"),
    ).pack()

    canvas = tk.Canvas(
        box,
        width=w * CELL,
        height=h * CELL,
        bg="white",
        highlightthickness=1,
        highlightbackground="black",
    )
    canvas.pack()

    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            x1 = c * CELL
            y1 = r * CELL
            x2 = x1 + CELL
            y2 = y1 + CELL

            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=COLORS.get(value, "#FFFFFF"),
                outline="#555555",
            )


def draw_text_block(parent, title, lines):
    box = tk.Frame(parent)
    box.pack(side=tk.LEFT, padx=10, pady=8, anchor="n")

    tk.Label(
        box,
        text=title,
        font=("Arial", 11, "bold"),
    ).pack(anchor="w")

    text = tk.Text(
        box,
        width=70,
        height=18,
        font=("Consolas", 10),
        wrap=tk.NONE,
    )
    text.pack()

    text.insert("1.0", "\n".join(lines))
    text.config(state=tk.DISABLED)


# ============================================================
# MAIN
# ============================================================

def main():
    task_id = input("Task id: ").strip()
    task_id, task = load_task(task_id)

    root = tk.Tk()
    root.title(f"Task Story Observer - {task_id}")

    tk.Label(
        root,
        text=f"TASK STORY OBSERVER: {task_id}",
        font=("Arial", 16, "bold"),
    ).pack(pady=8)

    outer = tk.Frame(root)
    outer.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(outer)
    y_scroll = tk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
    x_scroll = tk.Scrollbar(outer, orient=tk.HORIZONTAL, command=canvas.xview)

    scroll_frame = tk.Frame(canvas)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(
        yscrollcommand=y_scroll.set,
        xscrollcommand=x_scroll.set,
    )

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    x_scroll.pack(side=tk.BOTTOM, fill=tk.X)

    for i, pair in enumerate(task.get("train", []), start=1):
        input_grid = pair["input"]
        output_grid = pair["output"]

        analysis = analyze_pair(input_grid, output_grid)

        tk.Label(
            scroll_frame,
            text=f"TRAIN PAIR {i}",
            font=("Arial", 14, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(14, 0))

        row1 = tk.Frame(scroll_frame)
        row1.pack(anchor="w")

        draw_grid(row1, "INPUT", input_grid)
        draw_grid(row1, "EXPECTED", output_grid)
        draw_text_block(row1, "STORY", analysis["story_lines"])

        row2 = tk.Frame(scroll_frame)
        row2.pack(anchor="w")

        input_lines = component_summary_lines(
            input_grid,
            analysis["input_components"],
            anchor_color=analysis["anchor_color"],
        )

        output_lines = component_summary_lines(
            output_grid,
            analysis["output_components"],
            anchor_color=analysis["anchor_color"],
        )

        draw_text_block(row2, "INPUT OBJECTS", input_lines)
        draw_text_block(row2, "OUTPUT OBJECTS", output_lines)

    for i, pair in enumerate(task.get("test", [])):
        input_grid = pair["input"]
        input_bg = most_common_color(input_grid)
        input_colors = colors_in_grid(input_grid)
        input_allowed = set(input_colors) - {input_bg}

        components = find_components_by_color_set(
            input_grid,
            input_allowed,
            background_color=input_bg,
        )

        tk.Label(
            scroll_frame,
            text=f"TEST PAIR {i}",
            font=("Arial", 14, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(20, 0))

        row = tk.Frame(scroll_frame)
        row.pack(anchor="w")

        draw_grid(row, "TEST INPUT", input_grid)

        test_lines = [
            f"test shape: {grid_shape(input_grid)}",
            f"background: {input_bg}",
            f"colors: {sorted(input_colors)}",
            f"object count: {len(components)}",
            "",
        ]
        test_lines.extend(
            component_summary_lines(
                input_grid,
                components,
                anchor_color=7,
            )
        )

        draw_text_block(row, "TEST OBJECTS", test_lines)

    root.mainloop()


if __name__ == "__main__":
    main()