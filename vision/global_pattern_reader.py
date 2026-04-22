from collections import Counter


def grid_h(grid):
    return len(grid)


def grid_w(grid):
    return len(grid[0]) if grid else 0


def count_colors(grid):
    ctr = Counter()
    for row in grid:
        for v in row:
            ctr[v] += 1
    return dict(ctr)


def unique_colors(grid):
    return sorted(set(v for row in grid for v in row))


def dominant_color(grid):
    counts = count_colors(grid)
    if not counts:
        return None
    return max(counts, key=counts.get)


def nonzero_bbox(grid):
    cells = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v != 0]
    if not cells:
        return None

    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return {
        "top": min(rs),
        "left": min(cs),
        "bottom": max(rs),
        "right": max(cs),
    }


def crop(grid, top, left, bottom, right):
    return [row[left:right + 1] for row in grid[top:bottom + 1]]


def repeated_row_count(grid):
    seen = Counter(tuple(row) for row in grid)
    return sum(count - 1 for count in seen.values() if count > 1)


def repeated_col_count(grid):
    h = grid_h(grid)
    w = grid_w(grid)
    cols = []
    for c in range(w):
        cols.append(tuple(grid[r][c] for r in range(h)))
    seen = Counter(cols)
    return sum(count - 1 for count in seen.values() if count > 1)


def horizontal_symmetry_score(grid):
    h = grid_h(grid)
    w = grid_w(grid)
    total = 0
    match = 0

    for r in range(h // 2):
        opposite = h - 1 - r
        for c in range(w):
            total += 1
            if grid[r][c] == grid[opposite][c]:
                match += 1

    return match / total if total else 0.0


def vertical_symmetry_score(grid):
    h = grid_h(grid)
    w = grid_w(grid)
    total = 0
    match = 0

    for c in range(w // 2):
        opposite = w - 1 - c
        for r in range(h):
            total += 1
            if grid[r][c] == grid[r][opposite]:
                match += 1

    return match / total if total else 0.0


def find_uniform_rectangles(grid, min_h=2, min_w=2, max_h=8, max_w=8):
    h = grid_h(grid)
    w = grid_w(grid)
    rects = []

    for top in range(h):
        for left in range(w):
            color = grid[top][left]

            for rh in range(min_h, min(max_h, h - top) + 1):
                for rw in range(min_w, min(max_w, w - left) + 1):
                    ok = True
                    for r in range(top, top + rh):
                        for c in range(left, left + rw):
                            if grid[r][c] != color:
                                ok = False
                                break
                        if not ok:
                            break

                    if ok:
                        rects.append({
                            "top": top,
                            "left": left,
                            "bottom": top + rh - 1,
                            "right": left + rw - 1,
                            "height": rh,
                            "width": rw,
                            "color": color,
                            "area": rh * rw,
                        })

    return rects


def keep_large_rectangles(rects, top_k=10):
    rects = sorted(rects, key=lambda x: x["area"], reverse=True)
    kept = []
    seen = set()

    for rect in rects:
        key = (rect["top"], rect["left"], rect["bottom"], rect["right"], rect["color"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(rect)
        if len(kept) >= top_k:
            break

    return kept


def find_dense_center_region(grid):
    bbox = nonzero_bbox(grid)
    if bbox is None:
        return None

    region = crop(grid, bbox["top"], bbox["left"], bbox["bottom"], bbox["right"])
    return {
        "bbox": bbox,
        "grid": region,
        "height": len(region),
        "width": len(region[0]) if region else 0,
    }


def read_global_pattern(grid):
    """
    Human-style first look at the whole grid.
    This does not solve the task.
    It just describes what stands out.
    """
    h = grid_h(grid)
    w = grid_w(grid)
    colors = unique_colors(grid)
    counts = count_colors(grid)
    dom = dominant_color(grid)
    bbox = nonzero_bbox(grid)
    repeated_rows = repeated_row_count(grid)
    repeated_cols = repeated_col_count(grid)
    h_sym = horizontal_symmetry_score(grid)
    v_sym = vertical_symmetry_score(grid)
    center_region = find_dense_center_region(grid)

    uniform_rects = find_uniform_rectangles(grid)
    uniform_rects = keep_large_rectangles(uniform_rects, top_k=10)

    summary = {
        "height": h,
        "width": w,
        "colors": colors,
        "color_counts": counts,
        "dominant_color": dom,
        "nonzero_bbox": bbox,
        "repeated_rows": repeated_rows,
        "repeated_cols": repeated_cols,
        "horizontal_symmetry_score": h_sym,
        "vertical_symmetry_score": v_sym,
        "center_region": center_region,
        "uniform_rectangles": uniform_rects,
    }

    return summary


def print_global_pattern_summary(summary):
    print("GLOBAL PATTERN SUMMARY:")
    print(f"  size                     : {summary['height']}x{summary['width']}")
    print(f"  colors                   : {summary['colors']}")
    print(f"  dominant_color           : {summary['dominant_color']}")
    print(f"  nonzero_bbox             : {summary['nonzero_bbox']}")
    print(f"  repeated_rows            : {summary['repeated_rows']}")
    print(f"  repeated_cols            : {summary['repeated_cols']}")
    print(f"  horizontal_symmetry      : {summary['horizontal_symmetry_score']:.3f}")
    print(f"  vertical_symmetry        : {summary['vertical_symmetry_score']:.3f}")

    center = summary["center_region"]
    if center is not None:
        print(f"  center_region_bbox       : {center['bbox']}")
        print(f"  center_region_shape      : {center['height']}x{center['width']}")
    else:
        print("  center_region            : None")

    print("  top_uniform_rectangles   :")
    for rect in summary["uniform_rectangles"][:5]:
        print(
            f"    color={rect['color']} "
            f"top={rect['top']} left={rect['left']} "
            f"bottom={rect['bottom']} right={rect['right']} "
            f"shape={rect['height']}x{rect['width']} area={rect['area']}"
        )