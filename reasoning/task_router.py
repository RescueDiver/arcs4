# reasoning/task_router.py

from reasoning.pattern_rule_engine import solve_pair_pattern_rule
from reasoning.region_rule_engine import solve_pair_region_rule
from reasoning.region_alignment_rule_engine_v2 import solve_pair_region_alignment_rule_v2
from reasoning.motif_layout_rule import solve_pair_motif_layout_rule
from reasoning.object_grid_rule import solve_pair_object_grid_rule
from reasoning.pattern_expansion_rule import solve_pair_pattern_expansion
from reasoning.seed_placement_expansion_rule import solve_pair_seed_placement_expansion


# ============================================================
# OPTIONAL TASK-LEVEL MULTI-SEED RULE
# ============================================================

try:
    from reasoning.multi_seed_composition_rule import (
        discover_multi_seed_composition_rule_for_task,
        apply_multi_seed_composition_rule,
        apply_multi_seed_composition_rule_for_train_pair,
    )
except ImportError:
    discover_multi_seed_composition_rule_for_task = None
    apply_multi_seed_composition_rule = None
    apply_multi_seed_composition_rule_for_train_pair = None


# Disabled from active router for now:
# from reasoning.object_rule_engine_v2 import solve_pair_object_rule_v2
# from reasoning.partition_rule_engine import solve_pair_partition_rule


# ============================================================
# BASIC HELPERS
# ============================================================

def grid_shape(grid):
    if grid is None:
        return 0, 0

    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def score_prediction(predicted, expected):
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
        score += 1000000

    return score


def shape_penalty(predicted, expected):
    if predicted is None or expected is None:
        return 0

    ph, pw = grid_shape(predicted)
    eh, ew = grid_shape(expected)

    if ph == eh and pw == ew:
        return 0

    return (abs(ph - eh) + abs(pw - ew)) * 20


def find_divider_column(grid):
    if grid is None:
        return None

    h, w = grid_shape(grid)

    if h == 0 or w == 0:
        return None

    for c in range(w):
        col_vals = [grid[r][c] for r in range(h)]

        if len(set(col_vals)) == 1 and col_vals[0] != 0:
            return c

    return None


def detect_task_type(input_grid, output_grid):
    in_h, in_w = grid_shape(input_grid)
    out_h, out_w = grid_shape(output_grid)

    divider_col = find_divider_column(input_grid)

    if divider_col is not None and divider_col > 0:
        return "motif_layout"

    if output_grid is not None and in_h == out_h and in_w == out_w:
        return "pattern_same_size"

    if output_grid is not None and out_h <= in_h and out_w <= in_w:
        return "region_extract"

    if output_grid is not None and (out_h > in_h or out_w > in_w):
        return "expansion"

    return "general"


def maybe_add_candidate(candidates, result, strategy_name, input_grid, output_grid):
    """
    Normalize one strategy result into the router's standard candidate format.
    """
    if result is None:
        return None

    pred = result.get("predicted")

    if pred is None:
        return None

    raw_score = result.get("score")

    if raw_score is None:
        raw_score = score_prediction(pred, output_grid)

    penalty = shape_penalty(pred, output_grid)
    adjusted = raw_score - penalty

    result["strategy"] = strategy_name
    result["raw_score"] = raw_score
    result["score"] = raw_score
    result["shape_penalty"] = penalty
    result["full_grid_penalty"] = 0
    result["adjusted_score"] = adjusted

    if output_grid is not None:
        result["exact"] = pred == output_grid
    else:
        result["exact"] = False

    candidates.append(result)
    return result


# ============================================================
# MULTI-SEED HELPERS
# ============================================================

def is_strong_multi_seed_result(result, train_pairs):
    """
    Decide whether the multi-seed task-level rule should override
    normal pair-level routing.

    Strong means:
      - it found one example per train pair
      - it found all train input seeds inside every output
      - every seed match ratio is perfect
      - preferably all train outputs are exact after reconstruction
    """
    if result is None:
        return False

    examples = result.get("examples", [])
    seed_count = len(train_pairs)

    if len(examples) != len(train_pairs):
        return False

    if not result.get("all_seeds_found", False):
        return False

    if not result.get("perfect_seed_matches", False):
        return False

    for ex in examples:
        placements = ex.get("placements", [])

        if len(placements) != seed_count:
            return False

        for p in placements:
            if p.get("ratio", 0) < 1.0:
                return False

    return True


def build_multi_seed_strategy_stats(multi_seed_rule, train_pairs):
    exact_count = multi_seed_rule.get("exact_count", 0)
    pair_count = multi_seed_rule.get("pair_count", len(train_pairs))
    total_score = multi_seed_rule.get("total_score", 0)
    confidence = multi_seed_rule.get("confidence", total_score)

    residual_rule = multi_seed_rule.get("residual_rule", {})
    residual_type = residual_rule.get("type")

    return {
        "multi_seed_composition_rule": {
            "pair_count": pair_count,
            "exact_count": exact_count,
            "total_adjusted_score": confidence,
            "total_raw_score": total_score,
            "all_seeds_found": multi_seed_rule.get("all_seeds_found", False),
            "perfect_seed_matches": multi_seed_rule.get("perfect_seed_matches", False),
            "residual_rule": residual_type,
        }
    }


# ============================================================
# DEBUG HELPERS
# ============================================================

def print_adjusted_debug(candidates):
    print("\nDEBUG ROUTER ADJUSTMENTS:")

    if not candidates:
        print("  No candidates.")
        return

    for result in candidates:
        pred = result.get("predicted")
        ph, pw = grid_shape(pred)

        print(
            f" {result.get('strategy'):<30} "
            f"raw={result.get('raw_score'):<7} "
            f"shape_penalty={result.get('shape_penalty'):<4} "
            f"full_grid_penalty={result.get('full_grid_penalty'):<4} "
            f"adjusted={result.get('adjusted_score'):<7} "
            f"pred_shape={ph}x{pw}"
        )


def debug_strategy_scores(
    result_seed_placement,
    result_pattern,
    result_region,
    result_motif_layout,
    result_region_alignment_v2,
    result_object_grid,
    result_pattern_expansion,
):
    print("\n=== STRATEGY SCORES ===")

    rows = [
        ("seed_placement_expansion_rule", result_seed_placement),
        ("pattern_rule", result_pattern),
        ("region_rule", result_region),
        ("motif_layout_rule", result_motif_layout),
        ("region_alignment_rule_v2", result_region_alignment_v2),
        ("object_grid_rule", result_object_grid),
        ("pattern_expansion_rule", result_pattern_expansion),
    ]

    for name, result in rows:
        if result is None:
            print(f"{name:<32}: None")
        else:
            print(f"{name:<32}: {result.get('score')}")


def debug_router_adjustments(
    result_seed_placement,
    result_pattern,
    result_region,
    result_motif_layout,
    result_region_alignment_v2,
    result_object_grid,
    result_pattern_expansion,
):
    print("\n=== ROUTER DECISION TABLE ===")

    rows = [
        ("seed_placement_expansion_rule", result_seed_placement),
        ("pattern_rule", result_pattern),
        ("region_rule", result_region),
        ("motif_layout_rule", result_motif_layout),
        ("region_alignment_rule_v2", result_region_alignment_v2),
        ("object_grid_rule", result_object_grid),
        ("pattern_expansion_rule", result_pattern_expansion),
    ]

    for name, result in rows:
        if result is None:
            print(f"{name:<32}: None")
            continue

        pred = result.get("predicted")
        ph, pw = grid_shape(pred)

        print(
            f"{name:<32} "
            f"raw={result.get('raw_score'):<7} "
            f"adj={result.get('adjusted_score'):<7} "
            f"shape_pen={result.get('shape_penalty'):<4} "
            f"full_pen={result.get('full_grid_penalty'):<4} "
            f"out={ph}x{pw}"
        )


# ============================================================
# ACTIVE PAIR ROUTER
# ============================================================

def get_all_strategy_results(input_grid, output_grid, debug=True):
    """
    Run normal pair-level strategies.

    This does NOT run multi_seed_composition_rule, because multi-seed is a
    task-level rule. It needs all train pairs together.
    """
    candidates = []

    task_type = detect_task_type(input_grid, output_grid)

    if debug:
        print(f"\n[TASK TYPE DETECTED] {task_type}")

    result_seed_placement = None
    result_pattern = None
    result_region = None
    result_motif_layout = None
    result_region_alignment_v2 = None
    result_object_grid = None
    result_pattern_expansion = None

    # --------------------------------------------------------
    # Seed placement expansion family
    # Finds where ONE input seed appears in a larger output.
    # --------------------------------------------------------
    if task_type == "expansion":
        result_seed_placement = maybe_add_candidate(
            candidates,
            solve_pair_seed_placement_expansion(input_grid, output_grid),
            "seed_placement_expansion_rule",
            input_grid,
            output_grid,
        )

        if result_seed_placement is not None and result_seed_placement.get("exact"):
            if debug:
                print("[ROUTER PRIORITY] seed_placement_expansion_rule exact match")

                debug_strategy_scores(
                    result_seed_placement,
                    result_pattern,
                    result_region,
                    result_motif_layout,
                    result_region_alignment_v2,
                    result_object_grid,
                    result_pattern_expansion,
                )

                debug_router_adjustments(
                    result_seed_placement,
                    result_pattern,
                    result_region,
                    result_motif_layout,
                    result_region_alignment_v2,
                    result_object_grid,
                    result_pattern_expansion,
                )

                print_adjusted_debug(candidates)

            return candidates

    # --------------------------------------------------------
    # Pattern family
    # Strong on same-size and simple grid edits.
    # --------------------------------------------------------
    if task_type in ["pattern_same_size", "general", "motif_layout", "expansion"]:
        result_pattern = maybe_add_candidate(
            candidates,
            solve_pair_pattern_rule(input_grid, output_grid),
            "pattern_rule",
            input_grid,
            output_grid,
        )

    # --------------------------------------------------------
    # Region families
    # Strong on crop/extract/alignment tasks.
    # --------------------------------------------------------
    if task_type in ["region_extract", "general", "motif_layout"]:
        result_region = maybe_add_candidate(
            candidates,
            solve_pair_region_rule(input_grid, output_grid),
            "region_rule",
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

    # --------------------------------------------------------
    # Motif family
    # Only run when divider/task detector says motif.
    # --------------------------------------------------------
    if task_type == "motif_layout":
        result_motif_layout = maybe_add_candidate(
            candidates,
            solve_pair_motif_layout_rule(input_grid, output_grid),
            "motif_layout_rule",
            input_grid,
            output_grid,
        )
    else:
        if debug:
            print("Skipping motif_layout_rule (task type not motif_layout)")

    # --------------------------------------------------------
    # Expansion baseline
    # Only run when output is larger than input.
    # --------------------------------------------------------
    if task_type == "expansion":
        result_pattern_expansion = maybe_add_candidate(
            candidates,
            solve_pair_pattern_expansion(input_grid, output_grid),
            "pattern_expansion_rule",
            input_grid,
            output_grid,
        )

    # --------------------------------------------------------
    # Object grid
    # Keep active but lower priority by score.
    # --------------------------------------------------------
    result_object_grid = maybe_add_candidate(
        candidates,
        solve_pair_object_grid_rule(input_grid, output_grid),
        "object_grid_rule",
        input_grid,
        output_grid,
    )

    if debug:
        debug_strategy_scores(
            result_seed_placement,
            result_pattern,
            result_region,
            result_motif_layout,
            result_region_alignment_v2,
            result_object_grid,
            result_pattern_expansion,
        )

        debug_router_adjustments(
            result_seed_placement,
            result_pattern,
            result_region,
            result_motif_layout,
            result_region_alignment_v2,
            result_object_grid,
            result_pattern_expansion,
        )

        print("\nOBJECT_GRID DEBUG RESULT:")

        if result_object_grid is None:
            print("  object_grid_rule: None")
        else:
            print(f"  strategy: {result_object_grid.get('strategy')}")
            print(f"  score   : {result_object_grid.get('score')}")
            print(f"  exact   : {result_object_grid.get('exact')}")

            pred = result_object_grid.get("predicted")

            if pred is not None:
                print(f"  shape   : {len(pred)}x{len(pred[0]) if pred else 0}")

        print_adjusted_debug(candidates)

    return candidates


def choose_best_result(candidates):
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda r: (
            1 if r.get("exact") else 0,
            r.get("adjusted_score", r.get("score", -10**9)),
        ),
    )


def solve_pair_with_multiple_strategies(input_grid, output_grid, debug=True):
    candidates = get_all_strategy_results(
        input_grid,
        output_grid,
        debug=debug,
    )

    return choose_best_result(candidates)


# ============================================================
# TASK-LEVEL STRATEGY PICKER
# ============================================================

def choose_task_level_strategy(train_pairs, debug=True):
    """
    Choose ONE strategy for the whole task.

    Important:
    multi_seed_composition_rule is checked first because it is not a
    normal pair-level strategy. It learns from all train inputs together.
    """

    # --------------------------------------------------------
    # 1. MULTI-SEED TASK-LEVEL OVERRIDE
    # --------------------------------------------------------
    if discover_multi_seed_composition_rule_for_task is not None:
        multi_seed_rule = discover_multi_seed_composition_rule_for_task(train_pairs)

        if is_strong_multi_seed_result(multi_seed_rule, train_pairs):
            if debug:
                print("\n[ROUTER OVERRIDE] Using multi_seed_composition_rule")
                print("[ROUTER OVERRIDE] Reason: all train seeds found with ratio=1.0")

                residual_rule = multi_seed_rule.get("residual_rule", {})
                print(
                    "[ROUTER OVERRIDE] Residual rule:",
                    residual_rule.get("type"),
                )

            stats = build_multi_seed_strategy_stats(
                multi_seed_rule,
                train_pairs,
            )

            return {
                "best_strategy": "multi_seed_composition_rule",
                "strategy_stats": stats,

                # Keep both names so old and new callers both work.
                "task_rule": multi_seed_rule,
                "rule": multi_seed_rule,
            }
    else:
        if debug:
            print("[MULTI-SEED WARNING] multi_seed_composition_rule import failed")

    # --------------------------------------------------------
    # 2. NORMAL PAIR-LEVEL STRATEGY RANKING
    # --------------------------------------------------------
    strategy_stats = {}

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        output_grid = pair["output"]

        candidates = get_all_strategy_results(
            input_grid,
            output_grid,
            debug=debug,
        )

        for result in candidates:
            strategy = result.get("strategy")

            if strategy is None:
                continue

            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    "pair_count": 0,
                    "exact_count": 0,
                    "total_adjusted_score": 0,
                    "total_raw_score": 0,
                }

            strategy_stats[strategy]["pair_count"] += 1
            strategy_stats[strategy]["total_adjusted_score"] += result.get(
                "adjusted_score",
                result.get("score", 0),
            )
            strategy_stats[strategy]["total_raw_score"] += result.get("score", 0)

            if result.get("exact"):
                strategy_stats[strategy]["exact_count"] += 1

    best_strategy = None
    best_key = None

    for strategy, stats in strategy_stats.items():
        key = (
            stats.get("exact_count", 0),
            stats.get("total_adjusted_score", 0),
            stats.get("pair_count", 0),
        )

        if best_key is None or key > best_key:
            best_key = key
            best_strategy = strategy

    return {
        "best_strategy": best_strategy,
        "strategy_stats": strategy_stats,
        "task_rule": None,
        "rule": None,
    }


# ============================================================
# FORCED PAIR-LEVEL STRATEGY SOLVER
# ============================================================

def solve_pair_with_forced_strategy(input_grid, output_grid, strategy_name):
    """
    Force a normal pair-level strategy.

    Note:
    multi_seed_composition_rule is NOT a normal pair-level strategy.
    It requires a learned task_rule, so use apply_task_rule_to_input()
    for multi-seed train/test predictions.
    """

    if strategy_name == "pattern_rule":
        result = solve_pair_pattern_rule(input_grid, output_grid)

    elif strategy_name == "region_rule":
        result = solve_pair_region_rule(input_grid, output_grid)

    elif strategy_name == "region_alignment_rule_v2":
        result = solve_pair_region_alignment_rule_v2(input_grid, output_grid)

    elif strategy_name == "motif_layout_rule":
        result = solve_pair_motif_layout_rule(input_grid, output_grid)

    elif strategy_name == "object_grid_rule":
        result = solve_pair_object_grid_rule(input_grid, output_grid)

    elif strategy_name == "pattern_expansion_rule":
        result = solve_pair_pattern_expansion(input_grid, output_grid)

    elif strategy_name == "seed_placement_expansion_rule":
        result = solve_pair_seed_placement_expansion(input_grid, output_grid)

    elif strategy_name == "multi_seed_composition_rule":
        print(
            "[FORCED STRATEGY ERROR] multi_seed_composition_rule needs a learned "
            "task_rule. Use apply_task_rule_to_input(...)."
        )
        return None

    else:
        print(f"[FORCED STRATEGY ERROR] Unknown strategy: {strategy_name}")
        return None

    candidates = []

    return maybe_add_candidate(
        candidates,
        result,
        strategy_name,
        input_grid,
        output_grid,
    )


# ============================================================
# APPLY LEARNED TASK RULE
# ============================================================

def apply_task_rule_to_input(
    strategy_name,
    task_rule,
    input_grid,
    expected_grid=None,
    pair_index=None,
):
    """
    Apply the chosen task-level rule.

    This is what run_oneV2.py should call after choose_task_level_strategy().

    For train pairs:
        pass pair_index so multi_seed can replay the learned train example.

    For test pairs:
        leave pair_index=None so multi_seed applies the learned rule to test.
    """

    # --------------------------------------------------------
    # Multi-seed task-level rule
    # --------------------------------------------------------
    if strategy_name == "multi_seed_composition_rule":
        if task_rule is None:
            print("[TASK RULE ERROR] Missing multi_seed task_rule.")
            return None

        # Train replay
        if pair_index is not None:
            if apply_multi_seed_composition_rule_for_train_pair is None:
                print(
                    "[TASK RULE ERROR] "
                    "apply_multi_seed_composition_rule_for_train_pair import failed."
                )
                return None

            return apply_multi_seed_composition_rule_for_train_pair(
                task_rule,
                pair_index,
            )

        # Test application
        if apply_multi_seed_composition_rule is None:
            print("[TASK RULE ERROR] apply_multi_seed_composition_rule import failed.")
            return None

        return apply_multi_seed_composition_rule(
            task_rule,
            input_grid,
        )

    # --------------------------------------------------------
    # Normal pair-level strategies
    # --------------------------------------------------------
    forced_result = solve_pair_with_forced_strategy(
        input_grid,
        expected_grid,
        strategy_name,
    )

    if forced_result is None:
        return None

    return forced_result.get("predicted")


def score_task_rule_prediction(
    strategy_name,
    task_rule,
    input_grid,
    expected_grid,
    pair_index=None,
):
    """
    Convenience wrapper:
    apply chosen rule, then score it.
    """
    predicted = apply_task_rule_to_input(
        strategy_name=strategy_name,
        task_rule=task_rule,
        input_grid=input_grid,
        expected_grid=expected_grid,
        pair_index=pair_index,
    )

    if predicted is None:
        return {
            "strategy": strategy_name,
            "predicted": None,
            "score": 0,
            "adjusted_score": 0,
            "exact": False,
        }

    score = score_prediction(predicted, expected_grid)

    return {
        "strategy": strategy_name,
        "predicted": predicted,
        "score": score,
        "adjusted_score": score,
        "exact": predicted == expected_grid,
    }