def same_shape(grid_a, grid_b):
    if len(grid_a) != len(grid_b):
        return False
    return all(len(r1) == len(r2) for r1, r2 in zip(grid_a, grid_b))


def get_changed_cells(inp, out):
    changes = []
    for r in range(len(inp)):
        for c in range(len(inp[0])):
            if inp[r][c] != out[r][c]:
                changes.append((r, c, inp[r][c], out[r][c]))
    return changes


def discover_row_difference_rule(train_pairs):
    corrections = {}

    for pair in train_pairs:
        inp = pair["input"]
        out = pair["output"]

        if not same_shape(inp, out):
            return None

        for r in range(len(inp)):
            for c in range(len(inp[0])):
                if inp[r][c] != out[r][c]:
                    key = (r, c, inp[r][c])
                    value = out[r][c]

                    if key in corrections and corrections[key] != value:
                        return None

                    corrections[key] = value

    if not corrections:
        return None

    return {
        "family": "identity_pattern_edit",
        "edit_type": "cell_correction",
        "params": {
            "corrections": corrections
        }
    }


def generate_identity_pattern_edit_candidates(train_pairs):
    candidate = discover_row_difference_rule(train_pairs)
    if candidate is None:
        return []
    return [candidate]


def execute_identity_pattern_edit(grid, rule):
    edit_type = rule.get("edit_type")
    params = rule.get("params", {})

    if edit_type == "cell_correction":
        corrections = params["corrections"]
        out = [row[:] for row in grid]

        h = len(out)
        w = len(out[0]) if h else 0

        for (r, c, val_in), val_out in corrections.items():
            if r < h and c < w and out[r][c] == val_in:
                out[r][c] = val_out

        return out

    return None