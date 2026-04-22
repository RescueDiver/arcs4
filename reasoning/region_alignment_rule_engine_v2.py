def solve_pair_region_alignment_rule_v2(input_grid, output_grid):
    from core.transforms import apply_transform
    from core.scoring import score_prediction

    best_result = None
    best_score = -10**9

    H = len(input_grid)
    W = len(input_grid[0]) if H else 0

    out_h = len(output_grid)
    out_w = len(output_grid[0]) if out_h else 0

    # ✅ DEFINE IT HERE
    transform_names = [
        "identity",
        "rotate_90",
        "rotate_180",
        "rotate_270",
        "flip_horizontal",
        "flip_vertical",
        "rotate_90_flip_horizontal",
        "rotate_270_flip_horizontal",
    ]

    for r in range(H - out_h + 1):
        for c in range(W - out_w + 1):
            region = [row[c:c + out_w] for row in input_grid[r:r + out_h]]

            for transform_name in transform_names:
                try:
                    transformed = apply_transform(region, transform_name)
                except Exception:
                    continue

                if len(transformed) != out_h or len(transformed[0]) != out_w:
                    continue

                score = score_prediction(transformed, output_grid)

                if score > best_score:
                    best_score = score
                    best_result = {
                        "predicted": transformed,
                        "score": score,
                        "exact": transformed == output_grid,
                        "transform": transform_name,
                        "strategy": "region_alignment_rule_v2",
                        "top": r,
                        "left": c,
                    }

    return best_result