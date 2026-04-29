from copy import deepcopy
from core.scoring import score_prediction


def grid_shape(grid):
    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def same_shape(grid_a, grid_b):
    if grid_a is None or grid_b is None:
        return False
    if len(grid_a) != len(grid_b):
        return False
    return all(len(r1) == len(r2) for r1, r2 in zip(grid_a, grid_b))


def row_is_uniform(row):
    return len(set(row)) == 1


def count_row_differences(row_a, row_b):
    if len(row_a) != len(row_b):
        return 10**9
    return sum(1 for a, b in zip(row_a, row_b) if a != b)


def patch_score(predicted, output_grid):
    score = score_prediction(predicted, output_grid)
    exact = predicted == output_grid
    if exact:
        score += 1_000_000
    return score, exact


def find_uniform_rows(grid):
    rows = []
    for r, row in enumerate(grid):
        if row_is_uniform(row):
            rows.append(r)
    return rows


def split_into_row_bands(grid):
    """
    Use full uniform rows as separators.
    Returns list of (start_row, end_row) for non-uniform bands.
    """
    h, _ = grid_shape(grid)
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


def build_repeating_row(unit, width):
    out = []
    for c in range(width):
        out.append(unit[c % len(unit)])
    return out


def best_repeating_unit_for_row(row):
    """
    Find the smallest repeating unit that reproduces the entire row.
    """
    w = len(row)

    for unit_w in range(1, w + 1):
        if w % unit_w != 0:
            continue

        unit = row[:unit_w]
        rebuilt = build_repeating_row(unit, w)

        if rebuilt == row:
            return unit

    return None


def best_consensus_unit(rows):
    """
    Given several sibling rows of same width, choose a repeating unit
    that best explains them collectively.

    Returns:
      unit, support_count, total_mismatches
    """
    if not rows:
        return None, 0, 10**9

    width = len(rows[0])
    candidates = []

    # collect exact repeating units from rows that already repeat cleanly
    for row in rows:
        unit = best_repeating_unit_for_row(row)
        if unit is not None:
            candidates.append(unit)

    if not candidates:
        return None, 0, 10**9

    best_unit = None
    best_support = -1
    best_total_mismatches = 10**9

    unique_candidates = []
    seen = set()
    for unit in candidates:
        key = tuple(unit)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(unit)

    for unit in unique_candidates:
        rebuilt = build_repeating_row(unit, width)

        support = 0
        total_mismatches = 0

        for row in rows:
            diffs = count_row_differences(row, rebuilt)
            total_mismatches += diffs
            if diffs == 0:
                support += 1

        if (support > best_support) or (
            support == best_support and total_mismatches < best_total_mismatches
        ):
            best_unit = unit
            best_support = support
            best_total_mismatches = total_mismatches

    return best_unit, best_support, best_total_mismatches


def extract_band_rows(grid, band_start, band_end):
    return [grid[r][:] for r in range(band_start, band_end + 1)]


def build_band_repair_candidate(grid):
    """
    Main idea:
    - split grid into horizontal non-uniform bands
    - inside each band, compare rows with sibling rows
    - if a repeating unit has strong support, repair outlier rows to match
    """
    out = deepcopy(grid)
    bands = split_into_row_bands(grid)

    if not bands:
        return None

    total_changed = 0
    total_support = 0
    repaired_any = False

    for band_index, (r0, r1) in enumerate(bands):
        band_rows = extract_band_rows(grid, r0, r1)

        # ignore tiny bands
        if len(band_rows) < 2:
            continue

        width = len(band_rows[0])

        # only use non-uniform rows as candidates for pattern repair
        candidate_rows = []
        candidate_positions = []

        for local_idx, row in enumerate(band_rows):
            if not row_is_uniform(row):
                candidate_rows.append(row)
                candidate_positions.append(local_idx)

        if len(candidate_rows) < 2:
            continue

        unit, support, total_mismatches = best_consensus_unit(candidate_rows)
        if unit is None:
            continue

        canonical = build_repeating_row(unit, width)

        # need at least 2 exact supporters or this is too weak
        if support < 2:
            continue

        total_support += support

        for local_idx, row in zip(candidate_positions, candidate_rows):
            diffs = count_row_differences(row, canonical)

            # repair only if row is close-ish to the canonical repeated row
            # and not already exact
            if diffs > 0 and diffs <= max(2, width // 6):
                global_r = r0 + local_idx
                out[global_r] = canonical[:]
                total_changed += diffs
                repaired_any = True

    return {
        "name": "row_band_pattern_repair",
        "predicted": out,
        "changed_cells": total_changed,
        "support": total_support,
        "repaired_any": repaired_any,
    }


def build_global_row_consensus_candidate(grid):
    """
    Backup mode:
    - group all non-uniform rows by row length
    - find repeated row patterns that occur more than once
    - repair near-matching outliers globally
    """
    out = deepcopy(grid)
    h, w = grid_shape(grid)

    non_uniform_rows = []
    row_positions = []

    for r in range(h):
        row = grid[r]
        if not row_is_uniform(row):
            non_uniform_rows.append(row)
            row_positions.append(r)

    if len(non_uniform_rows) < 2:
        return None

    unit, support, total_mismatches = best_consensus_unit(non_uniform_rows)
    if unit is None or support < 2:
        return None

    canonical = build_repeating_row(unit, w)

    total_changed = 0
    repaired_any = False

    for r, row in zip(row_positions, non_uniform_rows):
        diffs = count_row_differences(row, canonical)

        if diffs > 0 and diffs <= max(2, w // 6):
            out[r] = canonical[:]
            total_changed += diffs
            repaired_any = True

    return {
        "name": "global_row_pattern_repair",
        "predicted": out,
        "changed_cells": total_changed,
        "support": support,
        "repaired_any": repaired_any,
    }


def choose_unsupervised_candidate(candidates):
    valid = [c for c in candidates if c is not None]
    if not valid:
        return None

    valid.sort(
        key=lambda c: (
            1 if c["repaired_any"] else 0,
            c["support"],
            -c["changed_cells"],
        ),
        reverse=True,
    )
    return valid[0]


def choose_supervised_candidate(candidates, output_grid):
    best = None
    best_score = -10**9

    for candidate in candidates:
        if candidate is None:
            continue

        predicted = candidate["predicted"]
        score, exact = patch_score(predicted, output_grid)

        if score > best_score:
            best_score = score
            best = {
                "predicted": predicted,
                "score": score,
                "exact": exact,
                "mode": candidate["name"],
                "support": candidate["support"],
                "changed_cells": candidate["changed_cells"],
            }

    return best


def solve_pair_panel_pattern_rule(input_grid, output_grid):
    """
    Full-width repeating-row repair family.

    TRAIN:
      - build several full-grid repair candidates
      - choose best by score against output

    TEST:
      - choose best by repair/support heuristic
    """
    if input_grid is None:
        return None

    candidates = [
        build_band_repair_candidate(input_grid),
        build_global_row_consensus_candidate(input_grid),
    ]

    if output_grid is None:
        chosen = choose_unsupervised_candidate(candidates)
        if chosen is None:
            return None

        return {
            "predicted": chosen["predicted"],
            "score": 0,
            "exact": False,
            "mode": chosen["name"],
            "support": chosen["support"],
            "changed_cells": chosen["changed_cells"],
        }

    return choose_supervised_candidate(candidates, output_grid)