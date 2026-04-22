from collections import deque


def find_objects(grid):
    """
    Finds connected nonzero objects using 4-connectivity.
    Any touching nonzero cells belong to the same object,
    even if they have different colors.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    objects = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 or visited[r][c]:
                continue

            q = deque([(r, c)])
            visited[r][c] = True
            cells = []

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc

                    if 0 <= nr < rows and 0 <= nc < cols:
                        if not visited[nr][nc] and grid[nr][nc] != 0:
                            visited[nr][nc] = True
                            q.append((nr, nc))

            min_r = min(rr for rr, _ in cells)
            max_r = max(rr for rr, _ in cells)
            min_c = min(cc for _, cc in cells)
            max_c = max(cc for _, cc in cells)

            bbox = {
                "min_r": min_r,
                "max_r": max_r,
                "min_c": min_c,
                "max_c": max_c,
            }

            patch = [[0 for _ in range(max_c - min_c + 1)] for _ in range(max_r - min_r + 1)]
            color_set = set()

            for rr, cc in cells:
                value = grid[rr][cc]
                patch[rr - min_r][cc - min_c] = value
                if value != 0:
                    color_set.add(value)

            obj = {
                "cells": cells,
                "area": len(cells),
                "bbox": bbox,
                "height": max_r - min_r + 1,
                "width": max_c - min_c + 1,
                "patch": patch,
                "color_set": sorted(color_set),
                "color_count": len(color_set),
            }
            objects.append(obj)

    return objects