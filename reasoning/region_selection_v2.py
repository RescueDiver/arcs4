from collections import Counter, deque
from core.scoring import score_prediction


def colors(grid):
    return set(v for row in grid for v in row)


def nonzero_colors(grid):
    return set(v for row in grid for v in row if v != 0)


def count_nonzero(grid):
    return sum(1 for row in grid for v in row if v != 0)


def crop(grid, top, left, bottom, right):
    return [row[left:right + 1] for row in grid[top:bottom + 1]]


def nonzero_bbox(grid):
    points = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v != 0]
    if not points:
        return None
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def trim_zero_margins(grid):
    if not grid or not grid[0]:
        return grid

    top = 0
    bottom = len(grid) - 1
    left = 0
    right = len(grid[0]) - 1

    while top <= bottom and all(v == 0 for v in grid[top]):
        top += 1

    while bottom >= top and all(v == 0 for v in grid[bottom]):
        bottom -= 1

    while left <= right and all(grid[r][left] == 0 for r in range(top, bottom + 1)):
        left += 1

    while right >= left and all(grid[r][right] == 0 for r in range(top, bottom + 1)):
        right -= 1

    if top > bottom or left > right:
        return [[0]]

    return [row[left:right + 1] for row in grid[top:bottom + 1]]


def trim_uniform_border(grid):
    if not grid or not grid[0]:
        return grid

    changed = True
    current = [row[:] for row in grid]

    while changed:
        changed = False
        h = len(current)
        w = len(current[0]) if h else 0

        if h <= 1 or w <= 1:
            break

        top_row = current[0]
        bottom_row = current[-1]
        left_col = [current[r][0] for r in range(h)]
        right_col = [current[r][-1] for r in range(h)]

        if len(set(top_row)) == 1:
            current = current[1:]
            changed = True
            continue

        if len(set(bottom_row)) == 1:
            current = current[:-1]
            changed = True
            continue

        if len(set(left_col)) == 1:
            current = [row[1:] for row in current]
            changed = True
            continue

        if len(set(right_col)) == 1:
            current = [row[:-1] for row in current]
            changed = True
            continue

    return current if current else [[0]]


def normalize_variants(grid):
    variants = [
        ("raw", grid),
        ("zero_trim", trim_zero_margins(grid)),
        ("border_trim", trim_uniform_border(grid)),
        ("zero_then_border", trim_uniform_border(trim_zero_margins(grid))),
        ("border_then_zero", trim_zero_margins(trim_uniform_border(grid))),
    ]

    deduped = []
    seen = set()

    for name, variant in variants:
        if not variant or not variant[0]:
            continue
        key = tuple(tuple(row) for row in variant)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, variant))

    return deduped


def neighbors4(r, c, h, w):
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def connected_components_nonzero(grid):
    h = len(grid)
    w = len(grid[0]) if h else 0
    seen = [[False] * w for _ in range(h)]
    comps = []

    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 or seen[r][c]:
                continue

            q = deque([(r, c)])
            seen[r][c] = True
            cells = []

            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))

                for nr, nc in neighbors4(rr, cc, h, w):
                    if not seen[nr][nc] and grid[nr][nc] != 0:
                        seen[nr][nc] = True
                        q.append((nr, nc))

            comps.append(cells)

    return comps


def bbox_of_cells(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def generate_candidate_regions_v2(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    candidates = []

    # 1) full nonzero bbox
    bbox = nonzero_bbox(grid)
    if bbox is not None:
        t, l, b, r = bbox
        candidates.append({
            "source": "full_nonzero_bbox",
            "top": t,
            "left": l,
            "height": b - t + 1,
            "width": r - l + 1,
            "grid": crop(grid, t, l, b, r),
        })

    # 2) connected components
    comps = connected_components_nonzero(grid)
    comp_boxes = []
    for i, comp in enumerate(comps):
        t, l, b, r = bbox_of_cells(comp)
        comp_boxes.append((t, l, b, r))
        candidates.append({
            "source": f"component_{i}",
            "top": t,
            "left": l,
            "height": b - t + 1,
            "width": r - l + 1,
            "grid": crop(grid, t, l, b, r),
        })

    # 3) merged component boxes
    for i in range(len(comp_boxes)):
        for j in range(i + 1, len(comp_boxes)):
            a = comp_boxes[i]
            b = comp_boxes[j]
            t = min(a[0], b[0])
            l = min(a[1], b[1])
            bb = max(a[2], b[2])
            rr = max(a[3], b[3])
            candidates.append({
                "source": f"merged_component_{i}_{j}",
                "top": t,
                "left": l,
                "height": bb - t + 1,
                "width": rr - l + 1,
                "grid": crop(grid, t, l, bb, rr),
            })

    # 4) color-specific bbox
    for color in sorted(nonzero_colors(grid)):
        cells = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == color]
        if not cells:
            continue
        t, l, b, r = bbox_of_cells(cells)
        candidates.append({
            "source": f"color_bbox_{color}",
            "top": t,
            "left": l,
            "height": b - t + 1,
            "width": r - l + 1,
            "grid": crop(grid, t, l, b, r),
        })

    # 5) row bands
    nonempty_rows = [r for r in range(rows) if any(grid[r][c] != 0 for c in range(cols))]
    if nonempty_rows:
        start = nonempty_rows[0]
        prev = nonempty_rows[0]
        band_id = 0
        for r in nonempty_rows[1:]:
            if r == prev + 1:
                prev = r
            else:
                candidates.append({
                    "source": f"row_band_{band_id}",
                    "top": start,
                    "left": 0,
                    "height": prev - start + 1,
                    "width": cols,
                    "grid": crop(grid, start, 0, prev, cols - 1),
                })
                band_id += 1
                start = r
                prev = r
        candidates.append({
            "source": f"row_band_{band_id}",
            "top": start,
            "left": 0,
            "height": prev - start + 1,
            "width": cols,
            "grid": crop(grid, start, 0, prev, cols - 1),
        })

    # 6) col bands
    nonempty_cols = [c for c in range(cols) if any(grid[r][c] != 0 for r in range(rows))]
    if nonempty_cols:
        start = nonempty_cols[0]
        prev = nonempty_cols[0]
        band_id = 0
        for c in nonempty_cols[1:]:
            if c == prev + 1:
                prev = c
            else:
                candidates.append({
                    "source": f"col_band_{band_id}",
                    "top": 0,
                    "left": start,
                    "height": rows,
                    "width": prev - start + 1,
                    "grid": crop(grid, 0, start, rows - 1, prev),
                })
                band_id += 1
                start = c
                prev = c
        candidates.append({
            "source": f"col_band_{band_id}",
            "top": 0,
            "left": start,
            "height": rows,
            "width": prev - start + 1,
            "grid": crop(grid, 0, start, rows - 1, prev),
        })

    # 7) full grid fallback
    candidates.append({
        "source": "full_grid",
        "top": 0,
        "left": 0,
        "height": rows,
        "width": cols,
        "grid": [row[:] for row in grid],
    })

    # dedupe by content + bbox
    deduped = []
    seen = set()
    for cand in candidates:
        key = (
            cand["top"],
            cand["left"],
            cand["height"],
            cand["width"],
            tuple(tuple(row) for row in cand["grid"])
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cand)

    return deduped


def region_features(grid):
    h = len(grid)
    w = len(grid[0]) if h else 0
    area = h * w
    nz = count_nonzero(grid)
    nz_ratio = nz / area if area else 0.0
    nz_colors = nonzero_colors(grid)

    tight = trim_zero_margins(grid)
    th = len(tight)
    tw = len(tight[0]) if th else 0
    tight_area = th * tw if th and tw else 1
    tight_fill = nz / tight_area if tight_area else 0.0

    counts = Counter(v for row in grid for v in row if v != 0)
    dominant_ratio = (max(counts.values()) / sum(counts.values())) if counts else 0.0

    return {
        "h": h,
        "w": w,
        "area": area,
        "nz": nz,
        "nz_ratio": nz_ratio,
        "tight_fill": tight_fill,
        "num_colors": len(nz_colors),
        "dominant_ratio": dominant_ratio,
        "aspect": (h / w) if w else 999.0,
    }


def pre_rank_region(region_grid, output_grid, input_grid):
    rf = region_features(region_grid)
    of = region_features(output_grid)

    input_h = len(input_grid)
    input_w = len(input_grid[0]) if input_h else 0

    score = 0.0

    # Dense / meaningful content
    score += rf["nz_ratio"] * 4.0
    score += rf["tight_fill"] * 4.0

    # Size closeness to output
    score -= abs(rf["h"] - of["h"]) * 1.5
    score -= abs(rf["w"] - of["w"]) * 1.5
    score -= abs(rf["area"] - of["area"]) * 0.2

    # Color closeness
    output_colors = nonzero_colors(output_grid)
    region_colors = nonzero_colors(region_grid)
    overlap = len(region_colors & output_colors)
    extra = len(region_colors - output_colors)
    missing = len(output_colors - region_colors)

    score += overlap * 2.0
    score -= extra * 1.5
    score -= missing * 0.5

    # Penalize full-grid overfiring
    if rf["h"] == input_h and rf["w"] == input_w:
        score -= 6.0

    # Penalize tiny junk
    if rf["area"] <= 2:
        score -= 5.0
    elif rf["area"] <= 4:
        score -= 2.5

    # Penalize nearly uniform junk
    if rf["num_colors"] <= 1 and rf["dominant_ratio"] > 0.9:
        score -= 2.0

    return score


def solve_pair_region_alignment_rule(input_grid, output_grid, top_k=12, debug=False):
    candidates = generate_candidate_regions_v2(input_grid)

    ranked = []

    for region in candidates:
        raw = region["grid"]

        for variant_name, grid in normalize_variants(raw):
            if not grid or not grid[0]:
                continue

            pre_score = pre_rank_region(grid, output_grid, input_grid)
            ranked.append({
                "strategy": "region_alignment_rule",
                "predicted": grid,
                "pre_score": pre_score,
                "region": region,
                "variant": variant_name,
            })

    ranked.sort(key=lambda x: x["pre_score"], reverse=True)
    ranked = ranked[:top_k]

    best = None
    best_score = -10**9
    output_colors = colors(output_grid)

    for item in ranked:
        grid = item["predicted"]

        # keep this filter, but apply it late, after ranking
        if len(grid) != len(output_grid) or len(grid[0]) != len(output_grid[0]):
            continue

        region_colors = colors(grid)
        if not region_colors.issubset(output_colors.union({0})):
            continue

        score = score_prediction(grid, output_grid)

        if score > best_score:
            best_score = score
            best = {
                "strategy": "region_alignment_rule",
                "predicted": grid,
                "score": score,
                "exact": grid == output_grid,
                "region": item["region"],
                "variant": item["variant"],
                "pre_score": item["pre_score"],
            }

    if debug:
        print("\n=== REGION ALIGNMENT V2 TOP RANKED ===")
        for i, item in enumerate(ranked[:5], 1):
            g = item["predicted"]
            print(
                f"{i}. src={item['region']['source']} "
                f"variant={item['variant']} "
                f"shape={len(g)}x{len(g[0]) if g else 0} "
                f"pre_score={item['pre_score']:.2f}"
            )

    return best