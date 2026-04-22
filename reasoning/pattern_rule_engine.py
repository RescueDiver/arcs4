def same_shape(grid_a, grid_b):
    if len(grid_a) != len(grid_b):
        return False
    return all(len(r1) == len(r2) for r1, r2 in zip(grid_a, grid_b))


def discover_cell_corrections(input_grid, output_grid):
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


def apply_pattern_rule(input_grid, rule):
    if rule is None:
        return None

    if rule["type"] == "cell_correction":
        out = [row[:] for row in input_grid]

        for r, c, old_val, new_val in rule["corrections"]:
            if r < len(out) and c < len(out[0]) and out[r][c] == old_val:
                out[r][c] = new_val

        return out

    return None


def solve_pair_pattern_rule(input_grid, output_grid):
    rule = discover_cell_corrections(input_grid, output_grid)
    if rule is None:
        return None

    predicted = apply_pattern_rule(input_grid, rule)
    exact = (predicted == output_grid)

    matches = 0
    if predicted is not None and same_shape(predicted, output_grid):
        for r in range(len(predicted)):
            for c in range(len(predicted[0])):
                if predicted[r][c] == output_grid[r][c]:
                    matches += 1

    return {
        "rule_type": "pattern_rule",
        "rule": rule,
        "predicted": predicted,
        "exact": exact,
        "score": matches + (1000000 if exact else 0),
    }