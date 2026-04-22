from reasoning.object_rule_engine import solve_pair_fc7_rule
from reasoning.object_rule_engine_v2 import solve_pair_object_rule_v2
from reasoning.pattern_rule_engine import solve_pair_pattern_rule
from reasoning.region_rule_engine import solve_pair_region_rule
from reasoning.partition_rule_engine import solve_pair_partition_rule
from reasoning.region_alignment_rule_engine import solve_pair_region_alignment_rule
from reasoning.mirror_repair_rule_engine import solve_pair_mirror_repair_rule
from debug.debug_utils import debug_strategy_scores, debug_router_adjustments
from reasoning.region_alignment_rule_engine_v2 import solve_pair_region_alignment_rule_v2
from reasoning.object_correspondence_rule_engine import solve_pair_object_correspondence_rule
from reasoning.object_projection_rule_engine import solve_pair_object_projection_rule
from reasoning.multi_object_projection_rule_engine import solve_pair_multi_object_projection_rule
from reasoning.motif_layout_rule import solve_pair_motif_layout_rule


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

    # Common object-style payload
    obj = result.get("selected_object")
    if isinstance(obj, dict) and "bbox" in obj:
        return obj["bbox"]

    # Sometimes bbox is stored directly
    if "bbox" in result and isinstance(result["bbox"], dict):
        return result["bbox"]

    # Sometimes region-like naming appears
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
    This is especially important when object_fc7 returns a big internal crop.
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
            f"  {item['strategy']:<20} "
            f"raw={item.get('raw_score', None):<6} "
            f"shape_penalty={item.get('shape_penalty', 0):<4} "
            f"full_grid_penalty={item.get('full_grid_penalty', 0):<4} "
            f"adjusted={item.get('adjusted_score', None):<6} "
            f"pred_shape={ph}x{pw}"
        )


def get_all_strategy_results(input_grid, output_grid):
    """
    Runs every available strategy and returns all adjusted candidate results.
    """
    candidates = []

    result_fc7 = solve_pair_fc7_rule(input_grid, output_grid)
    if result_fc7 is not None:
        result_fc7["strategy"] = "object_fc7"
        result_fc7 = apply_router_adjustments(result_fc7, input_grid, output_grid)
        candidates.append(result_fc7)

    result_v2 = solve_pair_object_rule_v2(input_grid, output_grid)
    if result_v2 is not None:
        result_v2["strategy"] = "object_v2_nonframe"
        result_v2 = apply_router_adjustments(result_v2, input_grid, output_grid)
        candidates.append(result_v2)

    result_pattern = solve_pair_pattern_rule(input_grid, output_grid)
    if result_pattern is not None:
        result_pattern["strategy"] = "pattern_rule"
        result_pattern = apply_router_adjustments(result_pattern, input_grid, output_grid)
        candidates.append(result_pattern)


    result_partition = solve_pair_partition_rule(input_grid, output_grid)
    if result_partition is not None:
        result_partition["strategy"] = "partition_rule"
        result_partition = apply_router_adjustments(result_partition, input_grid, output_grid)
        candidates.append(result_partition)

    result_region = solve_pair_region_rule(input_grid, output_grid)
    if result_region is not None:
        result_region["strategy"] = "region_rule"
        result_region = apply_router_adjustments(result_region, input_grid, output_grid)
        candidates.append(result_region)

    result_motif_layout = solve_pair_motif_layout_rule(input_grid, output_grid)
    if result_motif_layout is not None:
        result_motif_layout["strategy"] = "motif_layout_rule"
        result_motif_layout = apply_router_adjustments(
            result_motif_layout, input_grid, output_grid
        )
        candidates.append(result_motif_layout)

    result_region_alignment = solve_pair_region_alignment_rule(input_grid, output_grid)
    if result_region_alignment is not None:
        result_region_alignment["strategy"] = "region_alignment_rule"
        result_region_alignment = apply_router_adjustments(
            result_region_alignment, input_grid, output_grid
        )
        candidates.append(result_region_alignment)

    result_region_alignment_v2 = solve_pair_region_alignment_rule_v2(input_grid, output_grid)
    if result_region_alignment_v2 is not None:
        result_region_alignment_v2["strategy"] = "region_alignment_rule_v2"
        result_region_alignment_v2 = apply_router_adjustments(
            result_region_alignment_v2, input_grid, output_grid
        )
        candidates.append(result_region_alignment_v2)

    result_mirror_repair = solve_pair_mirror_repair_rule(input_grid, output_grid)
    if result_mirror_repair is not None:
        result_mirror_repair["strategy"] = "mirror_repair_rule"
        result_mirror_repair = apply_router_adjustments(
            result_mirror_repair, input_grid, output_grid
        )
        candidates.append(result_mirror_repair)

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
    Keeps the old behavior: best adjusted score wins.
    """
    if not candidates:
        return None

    candidates.sort(key=lambda x: x["adjusted_score"], reverse=True)
    return candidates[0]


def solve_pair_with_multiple_strategies(input_grid, output_grid):
    candidates = get_all_strategy_results(input_grid, output_grid)
    return choose_best_candidate(candidates)

    result_fc7 = solve_pair_fc7_rule(input_grid, output_grid)
    if result_fc7 is not None:
        result_fc7["strategy"] = "object_fc7"
        result_fc7 = apply_router_adjustments(result_fc7, input_grid, output_grid)
        candidates.append(result_fc7)

    result_v2 = solve_pair_object_rule_v2(input_grid, output_grid)
    if result_v2 is not None:
        result_v2["strategy"] = "object_v2_nonframe"
        result_v2 = apply_router_adjustments(result_v2, input_grid, output_grid)
        candidates.append(result_v2)

    result_pattern = solve_pair_pattern_rule(input_grid, output_grid)
    if result_pattern is not None:
        result_pattern["strategy"] = "pattern_rule"
        result_pattern = apply_router_adjustments(result_pattern, input_grid, output_grid)
        candidates.append(result_pattern)

    result_region = solve_pair_region_rule(input_grid, output_grid)
    if result_region is not None:
        result_region["strategy"] = "region_rule"
        result_region = apply_router_adjustments(result_region, input_grid, output_grid)
        candidates.append(result_region)

    result_object_correspondence = solve_pair_object_correspondence_rule(input_grid, output_grid)
    if result_object_correspondence is not None:
        result_object_correspondence["strategy"] = "object_correspondence_rule"
        result_object_correspondence = apply_router_adjustments(
            result_object_correspondence, input_grid, output_grid
        )
        candidates.append(result_object_correspondence)

    result_object_projection = solve_pair_object_projection_rule(input_grid, output_grid)
    if result_object_projection is not None:
        result_object_projection["strategy"] = "object_projection_rule"
        result_object_projection = apply_router_adjustments(
            result_object_projection, input_grid, output_grid
        )
        candidates.append(result_object_projection)

    result_multi_object_projection = solve_pair_multi_object_projection_rule(input_grid, output_grid)
    if result_multi_object_projection is not None:
        result_multi_object_projection["strategy"] = "multi_object_projection_rule"
        result_multi_object_projection = apply_router_adjustments(
            result_multi_object_projection, input_grid, output_grid
        )
        candidates.append(result_multi_object_projection)

        result_partition = solve_pair_partition_rule(input_grid, output_grid)
    if result_partition is not None:
        result_partition["strategy"] = "partition_rule"
        result_partition = apply_router_adjustments(result_partition, input_grid, output_grid)
        candidates.append(result_partition)

    result_region_alignment = solve_pair_region_alignment_rule(input_grid, output_grid)
    if result_region_alignment is not None:
        result_region_alignment["strategy"] = "region_alignment_rule"
        result_region_alignment = apply_router_adjustments(result_region_alignment, input_grid, output_grid)
        candidates.append(result_region_alignment)

    result_region_alignment_v2 = solve_pair_region_alignment_rule_v2(input_grid, output_grid)
    if result_region_alignment_v2 is not None:
        result_region_alignment_v2["strategy"] = "region_alignment_rule_v2"
        result_region_alignment_v2 = apply_router_adjustments(result_region_alignment_v2, input_grid, output_grid)
        candidates.append(result_region_alignment_v2)

    result_mirror_repair = solve_pair_mirror_repair_rule(input_grid, output_grid)
    if result_mirror_repair is not None:
        result_mirror_repair["strategy"] = "mirror_repair_rule"
        result_mirror_repair = apply_router_adjustments(result_mirror_repair, input_grid, output_grid)
        candidates.append(result_mirror_repair)

    debug_strategy_scores(
        result_fc7,
        result_v2,
        result_object_correspondence,
        result_object_projection,
        result_multi_object_projection,
        result_pattern,
        result_region,
        result_partition,
        result_region_alignment,
        result_mirror_repair,
        result_region_alignment_v2,
    )

    debug_router_adjustments(
        result_fc7,
        result_v2,
        result_object_correspondence,
        result_object_projection,
        result_multi_object_projection,
        result_pattern,
        result_region,
        result_partition,
        result_region_alignment,
        result_mirror_repair,
        result_region_alignment_v2,
    )

    print_adjusted_debug(candidates)

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["adjusted_score"], reverse=True)
    return candidates[0]


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

    Returns:
        {
            "best_strategy": str or None,
            "strategy_stats": {
                strategy_name: {
                    "exact_count": int,
                    "total_adjusted_score": int,
                    "pair_count": int,
                    "results": [result_or_none, ...]
                }
            }
        }
    """
    strategy_stats = {}

    for pair in train_pairs:
        input_grid = pair["input"]
        output_grid = pair["output"]

        candidates = get_all_strategy_results(input_grid, output_grid)

        # Index this pair's candidates by strategy name
        by_strategy = {c["strategy"]: c for c in candidates}

        # Track all known strategies seen so far
        for strategy_name in by_strategy.keys():
            if strategy_name not in strategy_stats:
                strategy_stats[strategy_name] = {
                    "exact_count": 0,
                    "total_adjusted_score": 0,
                    "pair_count": 0,
                    "results": [],
                }

        # Update per-strategy stats for this pair
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
        return {"best_strategy": None, "strategy_stats": {}}

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