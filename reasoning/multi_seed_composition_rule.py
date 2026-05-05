# reasoning/multi_seed_composition_rule.py

from core.scoring import score_prediction


# ============================================================
# SETTINGS
# ============================================================

MIN_MATCH_RATIO = 0.90
MAX_MATCHES_PER_SEED = 25
MAX_TOTAL_CANDIDATES = 200


# ============================================================
# BASIC GRID HELPERS
# ============================================================

def grid_shape(grid):
    if grid is None:
        return 0, 0

    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def make_grid(h, w, fill=0):
    return [[fill for _ in range(w)] for _ in range(h)]


def most_common_color(grid):
    counts = {}

    for row in grid:
        for v in row:
            counts[v] = counts.get(v, 0) + 1

    if not counts:
        return 0

    return max(counts, key=counts.get)


# ============================================================
# TRANSFORMS
# ============================================================

def rotate_grid_90(grid):
    return [list(row) for row in zip(*grid[::-1])]


def rotate_grid_180(grid):
    return rotate_grid_90(rotate_grid_90(grid))


def rotate_grid_270(grid):
    return rotate_grid_90(rotate_grid_180(grid))


def flip_grid_horizontal(grid):
    return [row[::-1] for row in grid]


def flip_grid_vertical(grid):
    return grid[::-1]


def get_grid_transforms(grid):
    raw = [
        ("identity", grid),
        ("rotate_90", rotate_grid_90(grid)),
        ("rotate_180", rotate_grid_180(grid)),
        ("rotate_270", rotate_grid_270(grid)),
        ("flip_horizontal", flip_grid_horizontal(grid)),
        ("flip_vertical", flip_grid_vertical(grid)),
        ("rotate_90_flip_horizontal", flip_grid_horizontal(rotate_grid_90(grid))),
        ("rotate_270_flip_horizontal", flip_grid_horizontal(rotate_grid_270(grid))),
    ]

    seen = set()
    out = []

    for name, g in raw:
        key = tuple(tuple(row) for row in g)

        if key in seen:
            continue

        seen.add(key)
        out.append((name, g))

    return out


# ============================================================
# COLOR MATCHING
# ============================================================

def score_seed_against_region(seed_grid, expected_grid, top, left):
    """
    Compare one transformed seed against one region of expected output.

    Allows color remapping:
        seed color A -> expected color X
        seed color B -> expected color Y

    But mapping must stay consistent.
    """
    seed_h, seed_w = grid_shape(seed_grid)
    exp_h, exp_w = grid_shape(expected_grid)

    color_map = {}
    reverse_map = {}

    matched = 0
    total = 0

    for r in range(seed_h):
        for c in range(seed_w):
            rr = top + r
            cc = left + c

            if not (0 <= rr < exp_h and 0 <= cc < exp_w):
                continue

            sv = seed_grid[r][c]
            ev = expected_grid[rr][cc]

            total += 1

            if sv in color_map:
                if color_map[sv] == ev:
                    matched += 1
            else:
                if ev in reverse_map:
                    continue

                color_map[sv] = ev
                reverse_map[ev] = sv
                matched += 1

    if total == 0:
        return 0.0, 0, 0, {}

    return matched / total, matched, total, color_map


def recolor_seed(seed_grid, color_map):
    out = []

    for row in seed_grid:
        new_row = []
        for v in row:
            new_row.append(color_map.get(v, v))
        out.append(new_row)

    return out


# ============================================================
# FIND SEED PLACEMENTS
# ============================================================

def find_seed_candidates(seed_grid, expected_grid, seed_index):
    """
    Find possible placements of one seed inside expected output.
    """
    exp_h, exp_w = grid_shape(expected_grid)

    candidates = []

    for transform_name, transformed_seed in get_grid_transforms(seed_grid):
        seed_h, seed_w = grid_shape(transformed_seed)

        if seed_h > exp_h or seed_w > exp_w:
            continue

        for top in range(exp_h - seed_h + 1):
            for left in range(exp_w - seed_w + 1):
                ratio, matched, total, color_map = score_seed_against_region(
                    transformed_seed,
                    expected_grid,
                    top,
                    left,
                )

                if ratio >= MIN_MATCH_RATIO:
                    recolored_seed = recolor_seed(transformed_seed, color_map)

                    candidates.append({
                        "seed_index": seed_index,
                        "transform": transform_name,
                        "top": top,
                        "left": left,
                        "height": seed_h,
                        "width": seed_w,
                        "ratio": ratio,
                        "matched": matched,
                        "total": total,
                        "color_map": color_map,
                        "transformed_seed": transformed_seed,
                        "recolored_seed": recolored_seed,
                    })

    candidates.sort(
        key=lambda c: (
            c["ratio"],
            c["matched"],
            c["height"] * c["width"],
        ),
        reverse=True,
    )

    return candidates[:MAX_MATCHES_PER_SEED]


def find_all_seed_candidates(seed_library, expected_grid):
    """
    Search every train input seed inside one expected output.
    """
    all_candidates = []

    for seed_index, seed_grid in enumerate(seed_library, start=1):
        seed_candidates = find_seed_candidates(
            seed_grid,
            expected_grid,
            seed_index,
        )

        all_candidates.extend(seed_candidates)

    all_candidates.sort(
        key=lambda c: (
            c["ratio"],
            c["matched"],
            c["height"] * c["width"],
        ),
        reverse=True,
    )

    return all_candidates[:MAX_TOTAL_CANDIDATES]


def keep_best_candidate_per_seed(candidates):
    """
    Keep only the best placement for each seed.

    This task appears to use:
        each seed once

    Not:
        hundreds of repeated seed stamps
    """
    best_per_seed = {}

    for c in candidates:
        sid = c["seed_index"]

        if sid not in best_per_seed:
            best_per_seed[sid] = c
            continue

        old = best_per_seed[sid]

        new_key = (
            c["ratio"],
            c["matched"],
            c["height"] * c["width"],
        )

        old_key = (
            old["ratio"],
            old["matched"],
            old["height"] * old["width"],
        )

        if new_key > old_key:
            best_per_seed[sid] = c

    return list(best_per_seed.values())


# ============================================================
# COMPOSITION
# ============================================================

def strip_background(piece, fill_color):
    """
    Turn background cells into None.

    None means:
        do not write this cell during overlay.
    """
    cleaned = []

    for row in piece:
        new_row = []

        for val in row:
            if val == fill_color:
                new_row.append(None)
            else:
                new_row.append(val)

        cleaned.append(new_row)

    return cleaned


def overlay_piece(canvas, piece, top, left):
    """
    Overlay masked piece onto canvas.

    None cells do not write.
    Real color cells do write.
    """
    for r in range(len(piece)):
        for c in range(len(piece[0])):
            val = piece[r][c]

            if val is None:
                continue

            rr = top + r
            cc = left + c

            if 0 <= rr < len(canvas) and 0 <= cc < len(canvas[0]):
                canvas[rr][cc] = val

    return canvas


def reconstruct_from_seed_candidates(seed_library, expected_grid):
    """
    Rebuild expected output from one best placement per seed.

    Important:
        Earlier 200-placement version caused chaos.
        This version uses only the strongest placement for each seed.
    """
    out_h, out_w = grid_shape(expected_grid)
    fill = most_common_color(expected_grid)

    canvas = make_grid(out_h, out_w, fill)

    candidates = find_all_seed_candidates(seed_library, expected_grid)

    candidates = [
        c for c in candidates
        if c["ratio"] >= MIN_MATCH_RATIO
    ]

    candidates = keep_best_candidate_per_seed(candidates)

    candidates.sort(
        key=lambda c: (
            c["top"],
            c["left"],
            c["seed_index"],
        )
    )

    placed = []

    for candidate in candidates:
        clean_piece = strip_background(
            candidate["recolored_seed"],
            fill,
        )

        canvas = overlay_piece(
            canvas,
            clean_piece,
            candidate["top"],
            candidate["left"],
        )

        placed.append(candidate)

    score = score_prediction(canvas, expected_grid)
    exact = canvas == expected_grid

    return {
        "predicted": canvas,
        "score": score,
        "adjusted_score": score,
        "exact": exact,
        "placements": placed,
        "candidate_count": len(placed),
        "fill_color": fill,
    }


# ============================================================
# PAIR-LEVEL SOLVER
# ============================================================

def solve_pair_multi_seed_composition(input_grid, output_grid, seed_library=None):
    if input_grid is None or output_grid is None:
        return None

    if seed_library is None:
        seed_library = [input_grid]

    result = reconstruct_from_seed_candidates(
        seed_library,
        output_grid,
    )

    return {
        "strategy": "multi_seed_composition_rule",
        "predicted": result["predicted"],
        "score": result["score"],
        "adjusted_score": result["adjusted_score"],
        "exact": result["exact"],
        "mode": "one_best_placement_per_seed",
        "placements": result["placements"],
        "candidate_count": result["candidate_count"],
        "fill_color": result["fill_color"],
    }


# ============================================================
# TASK-LEVEL DISCOVERY
# ============================================================

def discover_multi_seed_composition_rule_for_task(train_pairs):
    if not train_pairs:
        return None

    seed_library = [pair["input"] for pair in train_pairs]

    examples = []
    total_score = 0
    exact_count = 0
    output_shape_counts = {}

    for pair_index, pair in enumerate(train_pairs):
        inp = pair["input"]
        out = pair["output"]

        result = solve_pair_multi_seed_composition(
            inp,
            out,
            seed_library=seed_library,
        )

        if result is None:
            examples.append({
                "pair_index": pair_index,
                "score": -100000,
                "exact": False,
                "placements": [],
                "placement_count": 0,
            })

            total_score -= 100000
            continue

        score = result["score"]
        exact = result["exact"]
        placements = result["placements"]

        total_score += score

        if exact:
            exact_count += 1

        out_shape = grid_shape(out)
        output_shape_counts[out_shape] = output_shape_counts.get(out_shape, 0) + 1

        examples.append({
            "pair_index": pair_index,
            "score": score,
            "exact": exact,
            "placements": placements,
            "placement_count": len(placements),
            "fill_color": result["fill_color"],
        })

    best_output_shape = sorted(
        output_shape_counts.keys(),
        key=lambda s: output_shape_counts[s],
        reverse=True,
    )[0]

    return {
        "family": "multi_seed_composition_rule",
        "rule_type": "compose_output_from_all_seed_inputs",
        "seed_library": seed_library,
        "output_shape": best_output_shape,
        "total_score": total_score,
        "exact_count": exact_count,
        "pair_count": len(train_pairs),
        "examples": examples,
    }


# ============================================================
# APPLY TO TEST
# ============================================================

def apply_multi_seed_composition_rule(rule, test_input):
    """
    First runnable test-time version.

    Uses the strongest learned training layout as a template,
    then overlays the test input at the first learned placement.
    """
    if rule is None:
        return None

    out_h, out_w = rule.get("output_shape", (20, 20))
    examples = rule.get("examples", [])

    if not examples:
        return make_grid(out_h, out_w, 0)

    best_example = max(
        examples,
        key=lambda ex: (
            ex.get("exact", False),
            ex.get("score", 0),
            ex.get("placement_count", 0),
        ),
    )

    fill = best_example.get("fill_color", 0)
    canvas = make_grid(out_h, out_w, fill)

    placements = best_example.get("placements", [])

    for placement in placements:
        piece = placement.get("recolored_seed")

        if piece is None:
            continue

        clean_piece = strip_background(piece, fill)

        canvas = overlay_piece(
            canvas,
            clean_piece,
            placement["top"],
            placement["left"],
        )

    if placements:
        anchor = placements[0]
        test_piece = strip_background(test_input, fill)

        canvas = overlay_piece(
            canvas,
            test_piece,
            anchor["top"],
            anchor["left"],
        )
    else:
        test_piece = strip_background(test_input, fill)
        canvas = overlay_piece(canvas, test_piece, 0, 0)

    return canvas