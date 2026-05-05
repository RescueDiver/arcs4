# reasoning/seed_placement_expansion_rule.py

from core.scoring import score_prediction


def grid_shape(grid):
    if grid is None:
        return 0, 0
    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def make_blank_grid(h, w, fill=0):
    return [[fill for _ in range(w)] for _ in range(h)]


def copy_grid(grid):
    return [row[:] for row in grid]


def place_grid(base, piece, top, left):
    """
    Place piece onto base at top,left.
    Only writes cells that fit inside base.
    """
    out = copy_grid(base)
    bh, bw = grid_shape(base)
    ph, pw = grid_shape(piece)

    for r in range(ph):
        for c in range(pw):
            rr = top + r
            cc = left + c

            if 0 <= rr < bh and 0 <= cc < bw:
                out[rr][cc] = piece[r][c]

    return out


def find_exact_seed_matches(seed_grid, expected_grid):
    """
    Find exact full seed inside expected.
    Color must match exactly.
    """
    seed_h, seed_w = grid_shape(seed_grid)
    exp_h, exp_w = grid_shape(expected_grid)

    matches = []

    for top in range(exp_h - seed_h + 1):
        for left in range(exp_w - seed_w + 1):
            ok = True

            for r in range(seed_h):
                for c in range(seed_w):
                    if expected_grid[top + r][left + c] != seed_grid[r][c]:
                        ok = False
                        break
                if not ok:
                    break

            if ok:
                matches.append((top, left))

    return matches


def infer_background_color(output_grid):
    """
    Pick the most common color in the expected output.
    This is usually the safest fill color for blank regions.
    """
    counts = {}

    for row in output_grid:
        for v in row:
            counts[v] = counts.get(v, 0) + 1

    if not counts:
        return 0

    return max(counts, key=counts.get)


def extract_pattern_from_output(output_grid):
    """
    Try to find a repeating tile pattern in the output.
    Very simple first version.
    """
    h, w = grid_shape(output_grid)

    for tile_h in range(2, h):
        for tile_w in range(2, w):

            ok = True

            for r in range(h):
                for c in range(w):
                    if output_grid[r][c] != output_grid[r % tile_h][c % tile_w]:
                        ok = False
                        break
                if not ok:
                    break

            if ok:
                return [row[:tile_w] for row in output_grid[:tile_h]]

    return None


def tile_pattern(pattern, out_h, out_w):
    ph = len(pattern)
    pw = len(pattern[0])

    out = []

    for r in range(out_h):
        row = []
        for c in range(out_w):
            row.append(pattern[r % ph][c % pw])
        out.append(row)

    return out


def build_seed_placement_candidate(input_grid, output_grid, top, left, fill_color=0):
    """
    Build candidate using a learned pattern from output_grid,
    then place the input seed.
    """
    out_h, out_w = grid_shape(output_grid)

    pattern = extract_pattern_from_output(output_grid)

    if pattern is not None:
        canvas = tile_pattern(pattern, out_h, out_w)
    else:
        canvas = make_blank_grid(out_h, out_w, fill_color)

    return place_grid(canvas, input_grid, top, left)


def solve_pair_seed_placement_expansion(input_grid, output_grid):
    """
    Pair-level version.

    This checks whether the input seed appears exactly inside the output.
    If it does, it builds a 20x20-style candidate using that placement.

    This is mainly for router scoring/debugging.
    """
    if input_grid is None or output_grid is None:
        return None

    out_h, out_w = grid_shape(output_grid)
    matches = find_exact_seed_matches(input_grid, output_grid)

    if not matches:
        return None

    fill_color = infer_background_color(output_grid)

    best = None

    for top, left in matches:
        out_h, out_w = grid_shape(output_grid)

        pred = build_seed_placement_candidate(
            input_grid,
            output_grid,
            top,
            left,
            fill_color=fill_color,
        )

        score = score_prediction(pred, output_grid)
        exact = pred == output_grid

        result = {
            "strategy": "seed_placement_expansion_rule",
            "predicted": pred,
            "score": score,
            "adjusted_score": score,
            "exact": exact,
            "mode": f"seed_at_{top}_{left}",
            "anchor": (top, left),
            "fill_color": fill_color,
        }

        if best is None or score > best["score"]:
            best = result

    return best


def discover_seed_placement_expansion_rule_for_task(train_pairs):
    """
    Task-level discovery.

    Learns:
        where the seed appears inside each expected output

    Then chooses the most common anchor placement as the task rule.
    """
    if not train_pairs:
        return None

    examples = []
    anchor_counts = {}
    anchor_scores = {}
    output_shapes = {}
    fill_counts = {}

    total_score = 0
    exact_count = 0

    for pair_index, pair in enumerate(train_pairs):
        inp = pair["input"]
        out = pair["output"]

        result = solve_pair_seed_placement_expansion(inp, out)

        if result is None:
            examples.append({
                "pair_index": pair_index,
                "found": False,
                "anchor": None,
                "score": -100000,
                "exact": False,
            })
            total_score -= 100000
            continue

        anchor = result["anchor"]
        out_shape = grid_shape(out)
        fill_color = result["fill_color"]

        anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1
        anchor_scores[anchor] = anchor_scores.get(anchor, 0) + result["score"]
        output_shapes[out_shape] = output_shapes.get(out_shape, 0) + 1
        fill_counts[fill_color] = fill_counts.get(fill_color, 0) + 1

        total_score += result["score"]

        if result["exact"]:
            exact_count += 1

        examples.append({
            "pair_index": pair_index,
            "found": True,
            "anchor": anchor,
            "score": result["score"],
            "exact": result["exact"],
            "fill_color": fill_color,
            "output_shape": out_shape,
        })

    valid_anchors = list(anchor_counts.keys())

    if not valid_anchors:
        return None

    best_anchor = sorted(
        valid_anchors,
        key=lambda a: (
            anchor_counts[a],
            anchor_scores[a],
        ),
        reverse=True,
    )[0]

    best_output_shape = sorted(
        output_shapes.keys(),
        key=lambda s: output_shapes[s],
        reverse=True,
    )[0]

    best_fill_color = sorted(
        fill_counts.keys(),
        key=lambda c: fill_counts[c],
        reverse=True,
    )[0]

    return {
        "family": "seed_placement_expansion_rule",
        "rule_type": "place_seed_in_larger_output",
        "anchor": best_anchor,
        "output_shape": best_output_shape,
        "fill_color": best_fill_color,
        "total_score": total_score,
        "exact_count": exact_count,
        "pair_count": len(train_pairs),
        "examples": examples,
        "anchor_counts": anchor_counts,
        "anchor_scores": anchor_scores,
    }


def apply_seed_placement_expansion_rule(rule, test_input):
    """
    Apply learned seed placement rule to a test input.
    """
    if rule is None:
        return None

    out_h, out_w = rule.get("output_shape", (20, 20))
    top, left = rule.get("anchor", (0, 0))
    fill_color = rule.get("fill_color", 0)

    return build_seed_placement_candidate(
        test_input,
        out_h,
        out_w,
        top,
        left,
        fill_color=fill_color,
    )