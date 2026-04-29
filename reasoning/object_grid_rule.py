from core.scoring import score_prediction


# ============================================================
# TASK-LEVEL MEMORY
# ============================================================

MODE_STATS = {}
LEARNED_MODE_CACHE = None


# ============================================================
# BASIC HELPERS
# ============================================================

def grid_shape(grid):
    if grid is None:
        return 0, 0
    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def paste_patch(out, patch, top, left):
    for r in range(len(patch)):
        for c in range(len(patch[0])):
            out[top + r][left + c] = patch[r][c]


def make_blank_tile(tile_h, tile_w):
    return {
        "top": 9999,
        "left": 9999,
        "bottom": 9999,
        "right": 9999,
        "height": tile_h,
        "width": tile_w,
        "area": 0,
        "color": 0,
        "grid": [[0 for _ in range(tile_w)] for _ in range(tile_h)],
    }


# ============================================================
# OBJECT EXTRACTION
# ============================================================

def extract_objects(grid):
    h, w = grid_shape(grid)
    visited = [[False for _ in range(w)] for _ in range(h)]
    objects = []

    def dfs(sr, sc, color):
        stack = [(sr, sc)]
        cells = []

        while stack:
            r, c = stack.pop()

            if r < 0 or r >= h or c < 0 or c >= w:
                continue
            if visited[r][c]:
                continue
            if grid[r][c] != color:
                continue

            visited[r][c] = True
            cells.append((r, c))

            stack.append((r - 1, c))
            stack.append((r + 1, c))
            stack.append((r, c - 1))
            stack.append((r, c + 1))

        return cells

    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 or visited[r][c]:
                continue

            color = grid[r][c]
            cells = dfs(r, c, color)

            min_r = min(rr for rr, _ in cells)
            max_r = max(rr for rr, _ in cells)
            min_c = min(cc for _, cc in cells)
            max_c = max(cc for _, cc in cells)

            patch = [
                grid[rr][min_c:max_c + 1]
                for rr in range(min_r, max_r + 1)
            ]

            objects.append({
                "top": min_r,
                "left": min_c,
                "bottom": max_r,
                "right": max_c,
                "height": max_r - min_r + 1,
                "width": max_c - min_c + 1,
                "area": len(cells),
                "color": color,
                "grid": patch,
            })

    return objects


# ============================================================
# TILE SELECTION
# ============================================================

def looks_like_tile(obj):
    return obj["height"] == 4 and obj["width"] == 4


def select_tile_objects(grid):
    objects = extract_objects(grid)
    tiles = [obj for obj in objects if looks_like_tile(obj)]
    tiles.sort(key=lambda o: (o["top"], o["left"]))

    print("\n[OBJECT_GRID DEBUG]")
    for i, obj in enumerate(tiles, start=1):
        print(
            f"{i}. top={obj['top']} left={obj['left']} "
            f"shape={obj['height']}x{obj['width']} "
            f"colors={[obj['color']]}"
        )

    return tiles


# ============================================================
# OUTPUT BUILDING
# ============================================================

def build_output_from_ordered_tiles(tiles, output_h, output_w, layout_mode="row"):
    if not tiles:
        return None

    tile_h = tiles[0]["height"]
    tile_w = tiles[0]["width"]

    if output_h % tile_h != 0 or output_w % tile_w != 0:
        return None

    rows = output_h // tile_h
    cols = output_w // tile_w

    out = [[0 for _ in range(output_w)] for _ in range(output_h)]

    for idx, obj in enumerate(tiles):
        if idx >= rows * cols:
            break

        if layout_mode == "row":
            row_slot = idx // cols
            col_slot = idx % cols
        else:
            row_slot = idx % rows
            col_slot = idx // rows

        paste_patch(
            out,
            obj["grid"],
            row_slot * tile_h,
            col_slot * tile_w,
        )

    return out


def pad_with_blank_tiles(tiles, output_h, output_w):
    if not tiles:
        return tiles

    tile_h = tiles[0]["height"]
    tile_w = tiles[0]["width"]

    if output_h % tile_h != 0 or output_w % tile_w != 0:
        return tiles

    rows = output_h // tile_h
    cols = output_w // tile_w
    slot_count = rows * cols

    padded = tiles[:]

    while len(padded) < slot_count:
        padded.append(make_blank_tile(tile_h, tile_w))

    return padded


# ============================================================
# ORDER MODES
# ============================================================

def column_bucket_order(tiles):
    if not tiles:
        return []

    buckets = []

    for obj in sorted(tiles, key=lambda o: o["left"]):
        placed = False

        for bucket in buckets:
            avg_left = sum(o["left"] for o in bucket) / len(bucket)

            if abs(avg_left - obj["left"]) <= 5:
                bucket.append(obj)
                placed = True
                break

        if not placed:
            buckets.append([obj])

    buckets.sort(key=lambda b: min(o["left"] for o in b))

    for bucket in buckets:
        bucket.sort(key=lambda o: o["top"])

    ordered = []
    max_len = max(len(b) for b in buckets)

    for i in range(max_len):
        for bucket in buckets:
            if i < len(bucket):
                ordered.append(bucket[i])

    return ordered


def cluster_columns(tiles, threshold=5):
    if not tiles:
        return []

    sorted_tiles = sorted(tiles, key=lambda o: o["left"])

    clusters = []
    current = [sorted_tiles[0]]

    for tile in sorted_tiles[1:]:
        if abs(tile["left"] - current[-1]["left"]) <= threshold:
            current.append(tile)
        else:
            clusters.append(current)
            current = [tile]

    clusters.append(current)
    return clusters


def cluster_band_order(tiles):
    if not tiles:
        return []

    clusters = cluster_columns(tiles, threshold=5)

    clusters.sort(key=lambda c: min(o["left"] for o in c))

    for cluster in clusters:
        cluster.sort(key=lambda o: o["top"])

    ordered = []
    max_len = max(len(c) for c in clusters)

    for i in range(max_len):
        for cluster in clusters:
            if i < len(cluster):
                ordered.append(cluster[i])

    return ordered


def vertical_band_order(tiles):
    if not tiles:
        return []

    sorted_tiles = sorted(tiles, key=lambda o: o["left"])
    mid = len(sorted_tiles) // 2

    left_band = sorted_tiles[:mid]
    right_band = sorted_tiles[mid:]

    left_band.sort(key=lambda o: o["top"])
    right_band.sort(key=lambda o: o["top"])

    ordered = []
    max_len = max(len(left_band), len(right_band))

    for i in range(max_len):
        if i < len(left_band):
            ordered.append(left_band[i])
        if i < len(right_band):
            ordered.append(right_band[i])

    return ordered


def generate_tile_orders(tiles, output_h=None, output_w=None):
    candidates = []
    seen = set()

    def add_order(name, ordered):
        key = tuple((o["top"], o["left"], o["color"]) for o in ordered)
        if key not in seen:
            seen.add(key)
            candidates.append((name, ordered))

    base = tiles[:]

    if output_h is not None and output_w is not None:
        base = pad_with_blank_tiles(base, output_h, output_w)

    add_order(
        "top_left_order",
        sorted(base, key=lambda o: (o["top"], o["left"]))
    )

    add_order(
        "left_top_order",
        sorted(base, key=lambda o: (o["left"], o["top"]))
    )

    top_order = sorted(base, key=lambda o: (o["top"], o["left"]))
    pair_reversed = []
    for i in range(0, len(top_order), 2):
        pair = top_order[i:i + 2]
        pair_reversed.extend(reversed(pair))
    add_order("top_order_pair_reversed", pair_reversed)

    add_order(
        "top_right_order",
        sorted(base, key=lambda o: (o["top"], -o["left"]))
    )

    add_order(
        "right_bias_order",
        sorted(base, key=lambda o: (-o["left"], o["top"]))
    )

    add_order(
        "column_bucket_order",
        column_bucket_order(base)
    )

    add_order(
        "vertical_band_order",
        vertical_band_order(base)
    )

    add_order(
        "cluster_band_order",
        cluster_band_order(base)
    )

    return candidates


def apply_order_mode(tiles, mode_name, output_h, output_w):
    for name, ordered in generate_tile_orders(
        tiles,
        output_h=output_h,
        output_w=output_w,
    ):
        if name == mode_name:
            return ordered

    return None


# ============================================================
# TASK MODE MEMORY
# ============================================================

def update_mode_memory(mode_name, score, exact):
    global MODE_STATS, LEARNED_MODE_CACHE

    if mode_name not in MODE_STATS:
        MODE_STATS[mode_name] = {
            "score": 0,
            "exact_count": 0,
            "pairs_seen": 0,
        }

    MODE_STATS[mode_name]["score"] += score
    MODE_STATS[mode_name]["pairs_seen"] += 1

    if exact:
        MODE_STATS[mode_name]["exact_count"] += 1

    LEARNED_MODE_CACHE = max(
        MODE_STATS,
        key=lambda m: (
            MODE_STATS[m]["exact_count"],
            MODE_STATS[m]["score"],
        )
    )


def print_mode_memory():
    print("\n[OBJECT_GRID TASK MODE MEMORY]")

    if not MODE_STATS:
        print("No mode stats yet.")
        return

    for mode_name, data in sorted(
        MODE_STATS.items(),
        key=lambda item: (-item[1]["exact_count"], -item[1]["score"])
    ):
        print(
            f"{mode_name}: "
            f"exact={data['exact_count']} "
            f"pairs_seen={data['pairs_seen']} "
            f"score={data['score']}"
        )

    print(f"Current learned mode: {LEARNED_MODE_CACHE}")


# ============================================================
# TRAIN MODE
# ============================================================

def is_solid_tile(tile):
    grid = tile["grid"]
    color = tile["color"]

    for row in grid:
        for v in row:
            if v != color:
                return False

    return True


def flip_grid_horizontal(grid):
    return [list(reversed(row)) for row in grid]


def build_solid_hollow_flip_output(tiles, output_h, output_w):

    if not tiles:
        return None

    tile_h = tiles[0]["height"]
    tile_w = tiles[0]["width"]

    if output_h % tile_h != 0 or output_w % tile_w != 0:
        return None

    rows = output_h // tile_h
    cols = output_w // tile_w

    if cols != 2:
        return None

    solid = []
    hollow = []

    for tile in tiles:
        if is_solid_tile(tile):
            solid.append(tile)
        else:
            hollow.append(tile)

    solid.sort(key=lambda o: o["top"])
    hollow.sort(key=lambda o: o["top"])

    blank = make_blank_tile(tile_h, tile_w)

    while len(solid) < rows:
        solid.append(blank)

    while len(hollow) < rows:
        hollow.append(blank)

    out = [[0 for _ in range(output_w)] for _ in range(output_h)]

    for r in range(rows):
        paste_patch(out, hollow[r]["grid"], r * tile_h, 0)
        paste_patch(out, solid[r]["grid"], r * tile_h, tile_w)

    return out


def solve_train_mode(input_grid, output_grid):
    output_h, output_w = grid_shape(output_grid)

    tiles = select_tile_objects(input_grid)

    if not tiles:
        return None

    best = None
    best_score = -10**9

    # 1. Try user-discovered solid/hollow flip rule first
    special_predicted = build_solid_hollow_flip_output(
        tiles,
        output_h,
        output_w,
    )

    if special_predicted is not None:
        special_score = score_prediction(special_predicted, output_grid)
        special_exact = special_predicted == output_grid

        if special_exact:
            special_score += 1_000_000

        best = {
            "strategy": "object_grid_rule",
            "predicted": special_predicted,
            "score": special_score,
            "exact": special_exact,
            "mode": "solid_hollow_flip",
            "tile_count": len(tiles),
            "output_shape": f"{output_h}x{output_w}",
        }

        best_score = special_score

    # 2. Try older fallback modes, but do NOT erase the special rule
    for order_name, ordered_tiles in generate_tile_orders(
        tiles,
        output_h=output_h,
        output_w=output_w,
    ):
        for layout_mode in ["row", "column"]:
            predicted = build_output_from_ordered_tiles(
                ordered_tiles,
                output_h,
                output_w,
                layout_mode=layout_mode,
            )

            if predicted is None:
                continue

            score = score_prediction(predicted, output_grid)
            exact = predicted == output_grid

            if exact:
                score += 1_000_000

            full_mode_name = f"{order_name}__{layout_mode}"

            update_mode_memory(full_mode_name, score, exact)

            if score > best_score:
                best_score = score
                best = {
                    "strategy": "object_grid_rule",
                    "predicted": predicted,
                    "score": score,
                    "exact": exact,
                    "mode": full_mode_name,
                    "tile_count": len(tiles),
                    "output_shape": f"{output_h}x{output_w}",
                }

    if best:
        print(
            f"[OBJECT_GRID BEST] "
            f"mode={best.get('mode')} "
            f"score={best.get('score')} "
            f"exact={best.get('exact')}"
        )
    else:
        print("[OBJECT_GRID BEST] None")

    print_mode_memory()

    return best


# ============================================================
# TEST MODE
# ============================================================

def infer_output_shape_from_tiles(tiles):
    if not tiles:
        return None

    tile_h = tiles[0]["height"]
    tile_w = tiles[0]["width"]

    # 🔥 split same way as your rule
    solid = [t for t in tiles if is_solid_tile(t)]
    hollow = [t for t in tiles if not is_solid_tile(t)]

    # 🔥 THIS is the key fix
    rows = max(len(solid), len(hollow))
    cols = 2

    return rows * tile_h, cols * tile_w


def solve_test_mode(input_grid):
    tiles = select_tile_objects(input_grid)

    if not tiles:
        return None

    shape = infer_output_shape_from_tiles(tiles)
    if shape is None:
        return None

    output_h, output_w = shape
    special_predicted = build_solid_hollow_flip_output(
        tiles,
        output_h,
        output_w,
    )

    if special_predicted is not None:
        return {
            "strategy": "object_grid_rule",
            "predicted": special_predicted,
            "score": 0,
            "exact": False,
            "mode": "test_solid_hollow_flip",
            "tile_count": len(tiles),
            "output_shape": f"{output_h}x{output_w}",
        }
    chosen_mode = LEARNED_MODE_CACHE

    if chosen_mode is None:
        order_mode = "column_bucket_order"
        layout_mode = "row"
    else:
        if "__" in chosen_mode:
            order_mode, layout_mode = chosen_mode.split("__", 1)
        else:
            order_mode = chosen_mode
            layout_mode = "row"

    ordered_tiles = apply_order_mode(
        tiles,
        order_mode,
        output_h,
        output_w,
    )

    if ordered_tiles is None:
        ordered_tiles = column_bucket_order(
            pad_with_blank_tiles(tiles, output_h, output_w)
        )
        layout_mode = "row"

    predicted = build_output_from_ordered_tiles(
        ordered_tiles,
        output_h,
        output_w,
        layout_mode=layout_mode,
    )

    if predicted is None:
        return None

    return {
        "strategy": "object_grid_rule",
        "predicted": predicted,
        "score": 0,
        "exact": False,
        "mode": chosen_mode or "test_column_bucket_order__row",
        "tile_count": len(tiles),
        "output_shape": f"{output_h}x{output_w}",
    }


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def solve_pair_object_grid_rule(input_grid, output_grid=None):
    if input_grid is None:
        return None

    if output_grid is None:
        return solve_test_mode(input_grid)

    return solve_train_mode(input_grid, output_grid)