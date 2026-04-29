# reasoning/pattern_rule_engine.py

def same_shape(grid_a, grid_b):
    if grid_a is None or grid_b is None:
        return False

    if len(grid_a) != len(grid_b):
        return False

    return all(len(r1) == len(r2) for r1, r2 in zip(grid_a, grid_b))


def count_matches(grid_a, grid_b):
    if not same_shape(grid_a, grid_b):
        return 0

    matches = 0
    for r in range(len(grid_a)):
        for c in range(len(grid_a[0])):
            if grid_a[r][c] == grid_b[r][c]:
                matches += 1
    return matches


def discover_cell_corrections(input_grid, output_grid):
    """
    Pair-specific fallback rule.
    Finds exact changed cells between input and output.
    """
    if output_grid is None:
        return None

    if not same_shape(input_grid, output_grid):
        return None

    corrections = []

    for r in range(len(input_grid)):
        for c in range(len(input_grid[0])):
            if input_grid[r][c] != output_grid[r][c]:
                corrections.append((r, c, input_grid[r][c], output_grid[r][c]))

    if not corrections:
        return None

    return {
        "type": "cell_correction",
        "corrections": corrections,
    }


def find_separator_rows(grid):
    """
    Rows where every cell is the same color.
    """
    rows = []
    for r, row in enumerate(grid):
        if len(set(row)) == 1:
            rows.append(r)
    return rows


def find_separator_cols(grid):
    """
    Columns where every cell is the same color.
    """
    h = len(grid)
    w = len(grid[0])

    cols = []
    for c in range(w):
        col = [grid[r][c] for r in range(h)]
        if len(set(col)) == 1:
            cols.append(c)
    return cols


def extract_non_separator_spans(size, separator_indices):
    """
    Example:
      size=10, separators=[0,5,9]
      -> spans [(1,4), (6,8)]
    """
    spans = []
    sep_set = set(separator_indices)

    start = None
    for i in range(size):
        if i not in sep_set:
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i - 1))
                start = None

    if start is not None:
        spans.append((start, size - 1))

    return spans


def crop_patch(grid, r0, r1, c0, c1):
    return [row[c0:c1 + 1] for row in grid[r0:r1 + 1]]


def patch_shape(patch):
    return len(patch), len(patch[0]) if patch else 0


def extract_tile_layout(grid):
    """
    Split a grid into interior tiles using full separator rows/cols.
    Returns tile spans and patches.
    """
    h = len(grid)
    w = len(grid[0])

    sep_rows = find_separator_rows(grid)
    sep_cols = find_separator_cols(grid)

    row_spans = extract_non_separator_spans(h, sep_rows)
    col_spans = extract_non_separator_spans(w, sep_cols)

    if not row_spans or not col_spans:
        return None

    tiles = {}
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            tiles[(ri, ci)] = crop_patch(grid, r0, r1, c0, c1)

    return {
        "row_spans": row_spans,
        "col_spans": col_spans,
        "tiles": tiles,
    }


def majority_patch(patches):
    """
    Choose the most common exact patch.
    """
    counts = {}
    for patch in patches:
        key = tuple(tuple(row) for row in patch)
        counts[key] = counts.get(key, 0) + 1

    best_key = max(counts, key=counts.get)
    return [list(row) for row in best_key]


def learn_tile_consensus_rule(train_pairs):
    """
    Learn a repeated tile-repair rule from outputs.

    For each output:
    - find separator rows/cols
    - extract interior tiles
    - for each tile position, learn the consensus patch
    """
    learned_row_spans = None
    learned_col_spans = None
    tile_examples = {}

    for pair in train_pairs:
        output_grid = pair["output"]
        layout = extract_tile_layout(output_grid)
        if layout is None:
            return None

        row_spans = layout["row_spans"]
        col_spans = layout["col_spans"]
        tiles = layout["tiles"]

        if learned_row_spans is None:
            learned_row_spans = row_spans
            learned_col_spans = col_spans
        else:
            if learned_row_spans != row_spans or learned_col_spans != col_spans:
                return None

        for key, patch in tiles.items():
            tile_examples.setdefault(key, []).append(patch)

    if learned_row_spans is None or learned_col_spans is None:
        return None

    canonical_tiles = {}
    for key, patches in tile_examples.items():
        canonical_tiles[key] = majority_patch(patches)

    return {
        "type": "tile_consensus_repair",
        "row_spans": learned_row_spans,
        "col_spans": learned_col_spans,
        "canonical_tiles": canonical_tiles,
    }


def paste_patch(out, patch, r0, c0):
    for rr in range(len(patch)):
        for cc in range(len(patch[0])):
            out[r0 + rr][c0 + cc] = patch[rr][cc]


def apply_pattern_rule(input_grid, rule):
    if rule is None:
        return None

    if rule["type"] == "cell_correction":
        out = [row[:] for row in input_grid]

        for r, c, old_val, new_val in rule["corrections"]:
            if r < len(out) and c < len(out[0]) and out[r][c] == old_val:
                out[r][c] = new_val

        return out

    if rule["type"] == "tile_consensus_repair":
        out = [row[:] for row in input_grid]

        row_spans = rule["row_spans"]
        col_spans = rule["col_spans"]
        canonical_tiles = rule["canonical_tiles"]

        for (ri, ci), patch in canonical_tiles.items():
            r0, r1 = row_spans[ri]
            c0, c1 = col_spans[ci]

            ph, pw = patch_shape(patch)
            if ph != (r1 - r0 + 1) or pw != (c1 - c0 + 1):
                return None

            paste_patch(out, patch, r0, c0)

        return out

    return None


def solve_pair_pattern_rule(input_grid, output_grid, learned_rule=None):
    """
    Supports:
    - train pair scoring
    - test prediction with learned task-level rule
    """
    if output_grid is None:
        if learned_rule is None:
            return None

        predicted = apply_pattern_rule(input_grid, learned_rule)
        if predicted is None:
            return None

        return {
            "rule_type": "pattern_rule",
            "rule": learned_rule,
            "predicted": predicted,
            "exact": False,
            "score": 0,
        }

    if learned_rule is not None:
        predicted = apply_pattern_rule(input_grid, learned_rule)
        if predicted is not None:
            exact = predicted == output_grid
            matches = count_matches(predicted, output_grid)

            return {
                "rule_type": "pattern_rule",
                "rule": learned_rule,
                "predicted": predicted,
                "exact": exact,
                "score": matches + (1000000 if exact else 0),
            }

    rule = discover_cell_corrections(input_grid, output_grid)
    if rule is None:
        return None

    predicted = apply_pattern_rule(input_grid, rule)
    exact = predicted == output_grid
    matches = count_matches(predicted, output_grid)

    return {
        "rule_type": "pattern_rule",
        "rule": rule,
        "predicted": predicted,
        "exact": exact,
        "score": matches + (1000000 if exact else 0),
    }


def find_uniform_rows(grid):
    rows = []
    for r, row in enumerate(grid):
        if len(set(row)) == 1:
            rows.append(r)
    return rows


def split_into_row_bands(grid):
    """
    Split grid into non-uniform row bands using full uniform rows as separators.

    Example:
      uniform rows at [0, 5, 10]
      bands -> [(1,4), (6,9)]
    """
    h = len(grid)
    uniform_rows = set(find_uniform_rows(grid))

    bands = []
    start = None

    for r in range(h):
        if r not in uniform_rows:
            if start is None:
                start = r
        else:
            if start is not None:
                bands.append((start, r - 1))
                start = None

    if start is not None:
        bands.append((start, h - 1))

    return bands


def learn_repeating_row_rule(train_pairs):
    """
    Learn row-wise repeating pattern rules by band position instead of exact grid shape.

    For each changed row:
    - find which non-uniform band it belongs to
    - store its offset inside that band
    - store the repeating unit of the output row
    """
    by_band_role = {}

    for pair in train_pairs:
        input_grid = pair["input"]
        output_grid = pair["output"]

        if not same_shape(input_grid, output_grid):
            return None

        bands = split_into_row_bands(output_grid)
        if not bands:
            return None

        for r in range(len(output_grid)):
            in_row = input_grid[r]
            out_row = output_grid[r]

            if in_row == out_row:
                continue

            band_index = None
            offset_in_band = None

            for bi, (r0, r1) in enumerate(bands):
                if r0 <= r <= r1:
                    band_index = bi
                    offset_in_band = r - r0
                    break

            if band_index is None:
                return None

            best_unit = None

            for unit_w in range(1, len(out_row) + 1):
                if len(out_row) % unit_w != 0:
                    continue

                unit = out_row[:unit_w]
                rebuilt = []
                for c in range(len(out_row)):
                    rebuilt.append(unit[c % unit_w])

                if rebuilt == out_row:
                    best_unit = unit
                    break

            if best_unit is None:
                return None

            role_key = (band_index, offset_in_band)

            if role_key not in by_band_role:
                by_band_role[role_key] = best_unit
            else:
                if by_band_role[role_key] != best_unit:
                    return None

    if not by_band_role:
        return None

    learned_rows = []
    for (band_index, offset_in_band), unit in sorted(by_band_role.items()):
        learned_rows.append({
            "band_index": band_index,
            "offset_in_band": offset_in_band,
            "unit": unit,
        })

    return {
        "type": "row_repeat_rule",
        "rows": learned_rows,
    }


def apply_pattern_rule(input_grid, rule):
    if rule is None:
        return None

    if rule["type"] == "cell_correction":
        out = [row[:] for row in input_grid]

        for r, c, old_val, new_val in rule["corrections"]:
            if r < len(out) and c < len(out[0]) and out[r][c] == old_val:
                out[r][c] = new_val

        return out

    if rule["type"] == "tile_consensus_repair":
        out = [row[:] for row in input_grid]

        row_spans = rule["row_spans"]
        col_spans = rule["col_spans"]
        canonical_tiles = rule["canonical_tiles"]

        for (ri, ci), patch in canonical_tiles.items():
            r0, r1 = row_spans[ri]
            c0, c1 = col_spans[ci]

            ph, pw = patch_shape(patch)
            if ph != (r1 - r0 + 1) or pw != (c1 - c0 + 1):
                return None

            paste_patch(out, patch, r0, c0)

        return out

    if rule["type"] == "row_repeat_rule":
        out = [row[:] for row in input_grid]

        bands = split_into_row_bands(input_grid)
        if not bands:
            return None

        for item in rule["rows"]:
            band_index = item["band_index"]
            offset_in_band = item["offset_in_band"]
            unit = item["unit"]

            if band_index >= len(bands):
                continue

            r0, r1 = bands[band_index]
            target_row = r0 + offset_in_band

            if target_row > r1 or target_row >= len(out):
                continue

            rebuilt = []
            for c in range(len(out[target_row])):
                rebuilt.append(unit[c % len(unit)])
            out[target_row] = rebuilt

        return out

    return None


def learn_pattern_rule_from_train_pairs(train_pairs):
    """
    Task-level learning entry point.
    Try simpler, size-agnostic rules first.
    """
    rule = learn_repeating_row_rule(train_pairs)
    if rule is not None:
        return rule

    rule = learn_tile_consensus_rule(train_pairs)
    if rule is not None:
        return rule

    return None