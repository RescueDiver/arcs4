# debug_anchor_repair.py

import json
import os
from collections import deque


def grid_shape(grid):
    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def print_grid(title, grid):
    h, w = grid_shape(grid)
    print()
    print(f"{title} (h={h}, w={w})")
    for row in grid:
        print(" ".join(str(x) for x in row))


def load_task(name):
    candidates = [
        name,
        os.path.join("../data", name),
        os.path.join("../data", f"{name}.json"),
        os.path.join("../data_failures", "extracted_tasks", name),
        os.path.join("../data_failures", "extracted_tasks", f"{name}.json"),
    ]

    for path in candidates:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f), path

    raise FileNotFoundError(name)


def color_counts(grid):
    counts = {}

    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1

    return counts


def most_common_color(grid):
    counts = color_counts(grid)
    return max(counts, key=counts.get)


def find_components(grid, wanted_colors):
    h, w = grid_shape(grid)
    seen = set()
    components = []

    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue

            if grid[r][c] not in wanted_colors:
                continue

            q = deque([(r, c)])
            seen.add((r, c))
            cells = []

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc, grid[cr][cc]))

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = cr + dr
                    nc = cc + dc

                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue

                    if (nr, nc) in seen:
                        continue

                    if grid[nr][nc] not in wanted_colors:
                        continue

                    seen.add((nr, nc))
                    q.append((nr, nc))

            rows = [x[0] for x in cells]
            cols = [x[1] for x in cells]
            colors = sorted(set(x[2] for x in cells))

            components.append({
                "cells": cells,
                "size": len(cells),
                "colors": colors,
                "top": min(rows),
                "bottom": max(rows),
                "left": min(cols),
                "right": max(cols),
                "height": max(rows) - min(rows) + 1,
                "width": max(cols) - min(cols) + 1,
            })

    components.sort(key=lambda x: x["size"], reverse=True)
    return components


def crop_box(grid, box):
    return [
        row[box["left"]:box["right"] + 1]
        for row in grid[box["top"]:box["bottom"] + 1]
    ]


def print_components(grid, label):
    bg = most_common_color(grid)
    counts = color_counts(grid)

    print()
    print("=" * 60)
    print(label)
    print("=" * 60)
    print("shape:", grid_shape(grid))
    print("counts:", counts)
    print("background guess:", bg)

    wanted = set(counts.keys())
    wanted.discard(bg)

    print("active colors:", sorted(wanted))

    comps = find_components(grid, wanted)

    for i, comp in enumerate(comps):
        print()
        print(f"component {i}")
        print("  size  :", comp["size"])
        print("  colors:", comp["colors"])
        print(
            "  box   :",
            f"top={comp['top']} bottom={comp['bottom']} "
            f"left={comp['left']} right={comp['right']} "
            f"h={comp['height']} w={comp['width']}",
        )

        preview = crop_box(grid, comp)
        print_grid("  crop", preview)


def main():
    task_name = input("Task id/file: ").strip()
    task, path = load_task(task_name)

    if "train" not in task:
        if task_name in task:
            task = task[task_name]
        elif len(task) == 1:
            task = next(iter(task.values()))
        else:
            print("Top-level keys:", list(task.keys())[:20])
            raise KeyError("Could not find train key or task id wrapper.")

    print("Loaded:", path)

    for pair_index, pair in enumerate(task["train"]):
        print()
        print("#" * 80)
        print(f"TRAIN PAIR {pair_index + 1}")
        print("#" * 80)

        print_grid("INPUT", pair["input"])
        print_components(pair["input"], "INPUT COMPONENTS")

        print_grid("EXPECTED", pair["output"])
        print_components(pair["output"], "EXPECTED COMPONENTS")


if __name__ == "__main__":
    main()