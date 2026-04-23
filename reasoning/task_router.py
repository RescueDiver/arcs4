from reasoning.object_rule_engine import solve_pair_fc7_rule
from reasoning.object_rule_engine_v2 import solve_pair_object_rule_v2
from reasoning.pattern_rule_engine import solve_pair_pattern_rule
from reasoning.region_rule_engine import solve_pair_region_rule
from reasoning.partition_rule_engine import solve_pair_partition_rule
from reasoning.region_alignment_rule_engine import solve_pair_region_alignment_rule
from reasoning.region_alignment_rule_engine_v2 import solve_pair_region_alignment_rule_v2
from reasoning.mirror_repair_rule_engine import solve_pair_mirror_repair_rule
from reasoning.motif_layout_rule import solve_pair_motif_layout_rule
from reasoning.noise_cleanup_rule import solve_pair_noise_cleanup

from debug.debug_utils import debug_strategy_scores, debug_router_adjustments


def grid_shape(grid):
    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def get_predicted_grid(result):
    if result is None:
        return None
    return result.get("predicted")


def get_selected_object_bbox(result):
    """
    Tries to read object bbox from different object-family result styles.
    """
    if result is None:
        return None

    obj = result.get("selected_object")
    if isinstance(obj, dict) and "bbox" in obj:
        return obj["bbox"]

    if "bbox" in result and isinstance(result["bbox"], dict):
        return result["bbox"]

    if "region_bbox" in result and isinstance(result["region_bbox"], dict):
        return result["region_bbox"]

    return None


def is_full_grid_bbox(bbox, input_grid):
    if bbox is None:
        return False

    h, w = grid_shape(input_grid)

    min_r = bbox.get("min_r", bbox.get("top"))
    max_r = bbox.get("max_r", bbox.get("bottom"))
    min_c = bbox.get("min_c", bbox.get("left"))
    max_c = bbox.get("max_c", bbox.get("right"))

    if min_r is None or max_r is None or min_c is None or max_c is None:
        return False

    return min_r == 0 and min_c == 0 and max_r == h - 1 and max_c == w - 1


def compute_shape_penalty(predicted, expected):
    """
    Strongly penalize families whose prediction shape is far from output shape.
    """
    if predicted is None or expected is None:
        return 0

    ph, pw = grid_shape(predicted)
    eh, ew = grid_shape(expected)

    dh = abs(ph - eh)
    dw = abs(pw - ew)

    return dh * 20 + dw * 20


def compute_full_grid_object_penalty(result, input_grid):
    """
    Penalize object families when they selected the entire grid as the 'object'.
    """
    if result is None:
        return 0

    bbox = get_selected_object_bbox(result)
    if bbox is not None and is_full_grid_bbox(bbox, input_grid):
        return 120

    return 0


def apply_router_adjustments(result, input_grid, output_grid):
    """
    Adds adjusted_score plus debug info.
    """
    if result is None:
        return None

    raw_score = result.get("score", -10**9)
    predicted = get_predicted_grid(result)

    shape_penalty = compute_shape_penalty(predicted, output_grid)
    full_grid_penalty = 0

    if result.get("strategy") in ("object_fc7", "object_v2_nonframe"):
        full_grid_penalty = compute_full_grid_object_penalty(result, input_grid)

    adjusted_score = raw_score - shape_penalty - full_grid_penalty

    result["raw_score"] = raw_score
    result["shape_penalty"] = shape_penalty
    result["full_grid_penalty"] = full_grid_penalty
    result["adjusted_score"] = adjusted_score

    return result


def print_adjusted_debug(candidates):
    print("\nDEBUG ROUTER ADJUSTMENTS:")
    for item in candidates:
        predicted = item.get("predicted")
        ph, pw = grid_shape(predicted) if predicted is not None else (0, 0)

        print(
            f" {item['strategy']:<24} "
            f"raw={item.get('raw_score', None):<6} "
            f"shape_penalty={item.get('shape_penalty', 0):<4} "
            f"full_grid_penalty={item.get('full_grid_penalty', 0):<4} "
            f"adjusted={item.get('adjusted_score', None):<6} "
            f"pred_shape={ph}x{pw}"
        )


def find_divider_column(grid):
    h = len(grid)
    w = len(grid[0])

    for c in range(w):
        column = [grid[r][c] for r in range(h)]
        if len(set(column)) == 1:
            return c

    return None


def maybe_add_candidate(candidates, result, strategy_name, input_grid, output_grid):
    """
    Standard helper:
    - attach strategy name
    - apply router adjustments
    - append if valid
    """
    if result is None:
        return None

    result["strategy"] = strategy_name
    result = apply_router_adjustments(result, input_grid, output_grid)
    candidates.append(result)
    return result


def get_all_strategy_results(input_grid, output_grid):
    """
    Runs every active strategy and returns all adjusted candidate results.
    """
    candidates = []

    result_fc7 = maybe_add_candidate(
        candidates,
        solve_pair_fc7_rule(input_grid, output_grid),
        "object_fc7",
        input_grid,
        output_grid,
    )

    result_v2 = maybe_add_candidate(
        candidates,
        solve_pair_object_rule_v2(input_grid, output_grid),
        "object_v2_nonframe",
        input_grid,
        output_grid,
    )

    result_pattern = maybe_add_candidate(
        candidates,
        solve_pair_pattern_rule(input_grid, output_grid),
        "pattern_rule",
        input_grid,
        output_grid,
    )

    result_partition = maybe_add_candidate(
        candidates,
        solve_pair_partition_rule(input_grid, output_grid),
        "partition_rule",
        input_grid,
        output_grid,
    )

    result_region = maybe_add_candidate(
        candidates,
        solve_pair_region_rule(input_grid, output_grid),
        "region_rule",
        input_grid,
        output_grid,
    )

    result_motif_layout = None
    divider_col = find_divider_column(input_grid)
    if divider_col is not None and divider_col > 0:
        result_motif_layout = maybe_add_candidate(
            candidates,
            solve_pair_motif_layout_rule(input_grid, output_grid),
            "motif_layout_rule",
            input_grid,
            output_grid,
        )
    else:
        print("Skipping motif_layout_rule (no divider)")

    result_noise = maybe_add_candidate(
        candidates,
        solve_pair_noise_cleanup(input_grid, output_grid),
        "noise_cleanup_rule",
        input_grid,
        output_grid,
    )

    result_region_alignment = maybe_add_candidate(
        candidates,
        solve_pair_region_alignment_rule(input_grid, output_grid),
        "region_alignment_rule",
        input_grid,
        output_grid,
    )

    result_region_alignment_v2 = maybe_add_candidate(
        candidates,
        solve_pair_region_alignment_rule_v2(input_grid, output_grid),
        "region_alignment_rule_v2",
        input_grid,
        output_grid,
    )

    result_mirror_repair = maybe_add_candidate(
        candidates,
        solve_pair_mirror_repair_rule(input_grid, output_grid),
        "mirror_repair_rule",
        input_grid,
        output_grid,
    )

    debug_strategy_scores(
        result_fc7,
        result_v2,
        result_pattern,
        result_region,
        result_motif_layout,
        result_partition,
        result_region_alignment,
        result_mirror_repair,
        result_region_alignment_v2,
    )

    debug_router_adjustments(
        result_fc7,
        result_v2,
        result_pattern,
        result_region,
        result_motif_layout,
        result_partition,
        result_region_alignment,
        result_mirror_repair,
        result_region_alignment_v2,
    )

    print_adjusted_debug(candidates)
    return candidates


def choose_best_candidate(candidates):
    """
    Best adjusted score wins.
    """
    if not candidates:
        return None

    candidates.sort(key=lambda x: x["adjusted_score"], reverse=True)
    return candidates[0]


def solve_pair_with_multiple_strategies(input_grid, output_grid):
    candidates = get_all_strategy_results(input_grid, output_grid)
    return choose_best_candidate(candidates)


def solve_pair_with_forced_strategy(input_grid, output_grid, forced_strategy):
    """
    Solve a pair using only the strategy selected at task level.
    """
    candidates = get_all_strategy_results(input_grid, output_grid)

    for candidate in candidates:
        if candidate.get("strategy") == forced_strategy:
            return candidate

    return None


def choose_task_level_strategy(train_pairs):
    """
    Evaluate all strategies across all train pairs and choose the most
    consistent family for the whole task.
    """
    strategy_stats = {}

    for pair in train_pairs:
        input_grid = pair["input"]
        output_grid = pair["output"]

        candidates = get_all_strategy_results(input_grid, output_grid)
        by_strategy = {c["strategy"]: c for c in candidates}

        for strategy_name in by_strategy.keys():
            if strategy_name not in strategy_stats:
                strategy_stats[strategy_name] = {
                    "exact_count": 0,
                    "total_adjusted_score": 0,
                    "pair_count": 0,
                    "results": [],
                }

        for strategy_name, stats in strategy_stats.items():
            result = by_strategy.get(strategy_name)

            if result is None:
                stats["results"].append(None)
                continue

            stats["pair_count"] += 1
            stats["total_adjusted_score"] += result.get("adjusted_score", -10**9)
            stats["results"].append(result)

            if result.get("exact", False):
                stats["exact_count"] += 1

    if not strategy_stats:
        return {
            "best_strategy": None,
            "strategy_stats": {},
        }

    ranked = sorted(
        strategy_stats.items(),
        key=lambda item: (
            item[1]["exact_count"],
            item[1]["total_adjusted_score"],
            item[1]["pair_count"],
        ),
        reverse=True,
    )

    best_strategy = ranked[0][0]

    return {
        "best_strategy": best_strategy,
        "strategy_stats": strategy_stats,
    }