# reasoning/local_constraint_rule.py


# ============================================================
# LOCAL CONSTRAINT RULE
# ============================================================
# Idea:
#   Learn small neighborhood rules from input -> expected output.
#
# Mental model:
#   A grid is like a language.
#   A 3x3 neighborhood is like a "word".
#   The output cell is what that word means.
#
# This first version:
#   1. Assumes the input is a seed in the top-left of the expected output.
#   2. Learns 3x3 neighborhood -> expected center cell.
#   3. Builds a 20x20 prediction.
#   4. Preserves the original input in the top-left.
#   5. Fills the rest using learned local rules when possible.
#   6. Falls back to simple tiling when no rule matches.
# ============================================================


def grid_shape(grid):
    if grid is None:
        return 0, 0

    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def most_common_color(grid):
    counts = {}

    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1

    if not counts:
        return 0

    return max(counts, key=counts.get)


def score_grid(predicted, expected):
    if predicted is None or expected is None:
        return 0

    ph, pw = grid_shape(predicted)
    eh, ew = grid_shape(expected)

    score = 0

    for r in range(min(ph, eh)):
        for c in range(min(pw, ew)):
            if predicted[r][c] == expected[r][c]:
                score += 1

    if predicted == expected:
        score += 10000

    return score


def get_cell_safe(grid, r, c, fallback):
    h, w = grid_shape(grid)

    if 0 <= r < h and 0 <= c < w:
        return grid[r][c]

    return fallback


def get_3x3_signature(grid, r, c, fallback):
    """
    Turns the 3x3 neighborhood around cell r,c into a tuple.

    Example shape:
        a b c
        d e f
        g h i

    Returns:
        (a,b,c,d,e,f,g,h,i)
    """
    values = []

    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            values.append(get_cell_safe(grid, r + dr, c + dc, fallback))

    return tuple(values)


def learn_local_rules_from_pair(input_grid, output_grid):
    """
    Learn rules from cells where input and output overlap.

    For every cell in the input:
        read input 3x3 neighborhood
        look at expected output cell at same position
        save neighborhood -> expected value
    """
    if input_grid is None or output_grid is None:
        return {}

    in_h, in_w = grid_shape(input_grid)
    out_h, out_w = grid_shape(output_grid)

    fallback = most_common_color(input_grid)
    rule_votes = {}

    max_r = min(in_h, out_h)
    max_c = min(in_w, out_w)

    for r in range(max_r):
        for c in range(max_c):
            signature = get_3x3_signature(input_grid, r, c, fallback)
            expected_value = output_grid[r][c]

            if signature not in rule_votes:
                rule_votes[signature] = {}

            rule_votes[signature][expected_value] = rule_votes[signature].get(expected_value, 0) + 1

    # Convert votes into final rule table
    rules = {}

    for signature, votes in rule_votes.items():
        best_value = max(votes, key=votes.get)
        rules[signature] = best_value

    return rules


def merge_rule_tables(rule_tables):
    """
    Merge multiple pair-level rule tables.

    If the same 3x3 signature appears in multiple train pairs,
    we let the most common output value win.
    """
    merged_votes = {}

    for rules in rule_tables:
        for signature, value in rules.items():
            if signature not in merged_votes:
                merged_votes[signature] = {}

            merged_votes[signature][value] = merged_votes[signature].get(value, 0) + 1

    merged_rules = {}

    for signature, votes in merged_votes.items():
        best_value = max(votes, key=votes.get)
        merged_rules[signature] = best_value

    return merged_rules


def make_raw_tile(input_grid, out_h, out_w):
    """
    Fallback prediction:
    repeat the input pattern.
    """
    in_h, in_w = grid_shape(input_grid)

    return [
        [input_grid[r % in_h][c % in_w] for c in range(out_w)]
        for r in range(out_h)
    ]


def apply_rules_iteratively(grid, rules, iterations=5):
    """
    Apply local rules multiple times (like cellular automata).
    This allows structure to propagate instead of just tile.
    """
    current = [row[:] for row in grid]

    for _ in range(iterations):
        next_grid = [row[:] for row in current]

        for r in range(len(current)):
            for c in range(len(current[0])):
                sig = get_3x3_signature(current, r, c, most_common_color(current))

                if sig in rules:
                    next_grid[r][c] = rules[sig]

        current = next_grid

    return current


def preserve_input_top_left(pred, input_grid):
    """
    Keep original input fixed in the top-left.
    This matters because for this task type, input often appears
    as a seed of the final output.
    """
    in_h, in_w = grid_shape(input_grid)
    pred_h, pred_w = grid_shape(pred)

    for r in range(min(in_h, pred_h)):
        for c in range(min(in_w, pred_w)):
            pred[r][c] = input_grid[r][c]


def make_local_constraint_prediction(input_grid, output_grid, rules):
    """
    Build a prediction using learned local constraints.
    """
    in_h, in_w = grid_shape(input_grid)

    if output_grid is not None:
        out_h, out_w = grid_shape(output_grid)
    else:
        # fallback for this task family
        out_h, out_w = 20, 20

    # Start with simple repeated seed.
    fallback_grid = make_raw_tile(input_grid, out_h, out_w)

    # Use fallback as initial seed.
    seed_grid = [row[:] for row in fallback_grid]

    # Preserve input in top-left.
    preserve_input_top_left(seed_grid, input_grid)

    # Apply learned local rules once.
    pred = apply_rules_iteratively(seed_grid, rules, iterations=5)

    # Preserve input again so local rules do not damage the known seed.
    preserve_input_top_left(pred, input_grid)

    return pred


def solve_pair_local_constraint_rule(input_grid, output_grid):
    """
    Pair-level solver.

    This version can learn only from this pair.
    That is enough for router comparison, but later we can build
    a task-level learned version using all train pairs.
    """
    if input_grid is None:
        return None

    if output_grid is None:
        return None

    rules = learn_local_rules_from_pair(input_grid, output_grid)

    if not rules:
        return None

    pred = make_local_constraint_prediction(input_grid, output_grid, rules)
    score = score_grid(pred, output_grid)

    return {
        "strategy": "local_constraint_rule",
        "mode": "pair_3x3_constraints",
        "predicted": pred,
        "score": score,
        "exact": pred == output_grid,
        "rules_learned": len(rules),
    }


def learn_local_rules_from_pair(input_grid, output_grid):
    """
    Learn local grammar rules from the EXPECTED output itself.

    Instead of only learning from the input overlap, this learns:
        expected 3x3 neighborhood -> expected center cell

    This is closer to treating the final 20x20 output as a valid
    grammar sample.
    """
    if output_grid is None:
        return {}

    fallback = most_common_color(output_grid)
    out_h, out_w = grid_shape(output_grid)

    rule_votes = {}

    for r in range(out_h):
        for c in range(out_w):
            signature = (
                r // 5,  # vertical region bucket
                c // 5,  # horizontal region bucket
                get_3x3_signature(output_grid, r, c, fallback)
            )
            expected_value = output_grid[r][c]

            if signature not in rule_votes:
                rule_votes[signature] = {}

            rule_votes[signature][expected_value] = (
                rule_votes[signature].get(expected_value, 0) + 1
            )

    rules = {}

    for signature, votes in rule_votes.items():
        best_value = max(votes, key=votes.get)
        rules[signature] = best_value

    return rules


def solve_pair_local_constraint_rule_with_learned_rule(input_grid, output_grid, learned_rule):
    """
    Use a task-level learned rule table.
    This is what we will use for test pairs later.
    """
    if input_grid is None:
        return None

    if learned_rule is None:
        return None

    rules = learned_rule.get("rules", {})
    if not rules:
        return None

    pred = make_local_constraint_prediction(input_grid, output_grid, rules)
    score = score_grid(pred, output_grid)

    return {
        "strategy": "local_constraint_rule",
        "mode": "task_3x3_constraints",
        "predicted": pred,
        "score": score,
        "exact": pred == output_grid if output_grid is not None else False,
        "rules_learned": len(rules),
    }