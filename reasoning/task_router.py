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


# ============================================================
# OPTIONAL TASK-LEVEL LEARNED REGION RULE
# ============================================================

try:
    from reasoning.learned_region_rule import (
        discover_learned_region_rule_for_task,
        apply_learned_region_rule,
        describe_learned_region_rule,
        debug_learned_region_choice,
    )
except ImportError:
    discover_learned_region_rule_for_task = None
    apply_learned_region_rule = None
    describe_learned_region_rule = None
    debug_learned_region_choice = None
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
        score += 1_000_000

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
# LEARNED REGION HELPERS
# ============================================================

def score_learned_region_rule_on_train(learned_rule, train_pairs):
    """
    Apply learned_region_rule to all train inputs and score against train outputs.

    Important:
        learned_region_rule must predict without using expected output.

    Debug:
        expected output is only used AFTER prediction so we can print
        whether the selected learned candidate was right or wrong.
    """
    if learned_rule is None or apply_learned_region_rule is None:
        return None

    total_score = 0
    exact_count = 0
    pair_count = 0
    results = []

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        output_grid = pair["output"]

        # ----------------------------------------------------
        # DEBUG PATH:
        # This prints what candidate learned_region_rule chose.
        # It does NOT use expected_grid to choose the candidate.
        # ----------------------------------------------------
        if debug_learned_region_choice is not None:
            chosen = debug_learned_region_choice(
                learned_rule,
                input_grid,
                expected_grid=output_grid,
                pair_index=pair_index,
            )

            if chosen is None:
                predicted = None
            else:
                predicted = chosen.get("predicted")

        # ----------------------------------------------------
        # NORMAL PATH:
        # Used if the debug helper failed to import.
        # ----------------------------------------------------
        else:
            predicted = apply_learned_region_rule(
                learned_rule,
                input_grid,
            )

        score = score_prediction(predicted, output_grid)
        exact = predicted == output_grid

        if exact:
            exact_count += 1

        pair_count += 1
        total_score += score

        results.append({
            "pair_index": pair_index,
            "predicted": predicted,
            "score": score,
            "exact": exact,
        })

    return {
        "strategy": "learned_region_rule",
        "task_rule": learned_rule,
        "rule": learned_rule,
        "pair_count": pair_count,
        "exact_count": exact_count,
        "total_raw_score": total_score,
        "total_adjusted_score": total_score,
        "results": results,
    }


def is_strong_learned_region_result(scored_rule):
    """
    Decide whether learned_region_rule is strong enough to become the
    chosen task-level strategy.

    First version is conservative:
        - must solve at least one train pair exactly
        - must produce predictions for all train pairs
    """
    if scored_rule is None:
        return False

    pair_count = scored_rule.get("pair_count", 0)
    exact_count = scored_rule.get("exact_count", 0)
    results = scored_rule.get("results", [])

    if pair_count == 0:
        return False

    if len(results) != pair_count:
        return False

    if exact_count == 0:
        return False

    for item in results:
        if item.get("predicted") is None:
            return False

    return True


def build_learned_region_strategy_stats(scored_rule):
    return {
        "learned_region_rule": {
            "pair_count": scored_rule.get("pair_count", 0),
            "exact_count": scored_rule.get("exact_count", 0),
            "total_adjusted_score": scored_rule.get("total_adjusted_score", 0),
            "total_raw_score": scored_rule.get("total_raw_score", 0),
            "pattern_type": scored_rule.get("task_rule", {}).get("pattern_type"),
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

    This does NOT run multi_seed_composition_rule or learned_region_rule,
    because those are task-level rules. They need all train pairs together.
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

    Task-level strategies:
        1. multi_seed_composition_rule
        2. learned_region_rule

    Fallback:
        normal pair-level strategy ranking
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
                "task_rule": multi_seed_rule,
                "rule": multi_seed_rule,
            }
    else:
        if debug:
            print("[MULTI-SEED WARNING] multi_seed_composition_rule import failed")

    # --------------------------------------------------------
    # 2. LEARNED REGION TASK-LEVEL RULE
    # --------------------------------------------------------
    learned_region_scored = None
    learned_region_rule = None

    if discover_learned_region_rule_for_task is not None:
        learned_region_rule = discover_learned_region_rule_for_task(train_pairs)

        if learned_region_rule is not None:
            learned_region_scored = score_learned_region_rule_on_train(
                learned_region_rule,
                train_pairs,
            )

            if debug:
                print("\n[LEARNED REGION DEBUG]")

                if describe_learned_region_rule is not None:
                    describe_learned_region_rule(learned_region_rule)

                if learned_region_scored is not None:
                    print("\nlearned_region_rule train replay")
                    print("-" * 60)
                    print(f"Exact count: {learned_region_scored.get('exact_count')}")
                    print(f"Pair count : {learned_region_scored.get('pair_count')}")
                    print(f"Total score: {learned_region_scored.get('total_raw_score')}")

            # ------------------------------------------------
            # TEMP DEBUG OVERRIDE
            # ------------------------------------------------
            # This is intentionally looser than the final rule.
            #
            # Why:
            #   region_rule is still pair-level and cannot run on test inputs.
            #   We need learned_region_rule to be chosen temporarily so that
            #   test predictions are not None.
            #
            # Do NOT judge main.py final score with this as the permanent router.
            # Once learned_region_rule train replay improves, we will make this
            # conservative again.
            # ------------------------------------------------
            if (
                is_strong_learned_region_result(learned_region_scored)
                and learned_region_scored.get("exact_count") == learned_region_scored.get("pair_count")
            ):

                stats = {
                    "learned_region_rule": {
                        "pair_count": learned_region_rule.get("train_pair_count", 0),
                        "exact_count": learned_region_rule.get("train_exact_count", 0),
                        "total_adjusted_score": learned_region_scored.get(
                            "total_adjusted_score",
                            0,
                        ) if learned_region_scored is not None else 0,
                        "total_raw_score": learned_region_scored.get(
                            "total_raw_score",
                            0,
                        ) if learned_region_scored is not None else 0,
                        "known_patterns": learned_region_rule.get("known_pattern_types"),
                        "known_shapes": learned_region_rule.get("known_output_shapes"),
                    }
                }

                return {
                    "best_strategy": "learned_region_rule",
                    "strategy_stats": stats,
                    "task_rule": learned_region_rule,
                    "rule": learned_region_rule,
                }

            # ------------------------------------------------
            # Conservative final-style override.
            # This currently probably will not trigger on 2d0172a1 because
            # learned replay is still weak, but we keep it here for later.
            # ------------------------------------------------
            if (
                is_strong_learned_region_result(learned_region_scored)
                and learned_region_scored.get("exact_count") == learned_region_scored.get("pair_count")
            ):
                if debug:
                    print("\n[ROUTER OVERRIDE] Using learned_region_rule")
                    print("[ROUTER OVERRIDE] Reason: learned rule solved all train pairs")

                stats = build_learned_region_strategy_stats(learned_region_scored)

                return {
                    "best_strategy": "learned_region_rule",
                    "strategy_stats": stats,
                    "task_rule": learned_region_rule,
                    "rule": learned_region_rule,
                }
    else:
        if debug:
            print("[LEARNED REGION WARNING] learned_region_rule import failed")

    # --------------------------------------------------------
    # 3. NORMAL PAIR-LEVEL STRATEGY RANKING
    # --------------------------------------------------------
    strategy_stats = {}

    # Include learned_region_rule in stats if it exists, but do not force it
    # unless the override above returned early.
    if learned_region_scored is not None:
        strategy_stats["learned_region_rule"] = {
            "pair_count": learned_region_scored.get("pair_count", 0),
            "exact_count": learned_region_scored.get("exact_count", 0),
            "total_adjusted_score": learned_region_scored.get("total_adjusted_score", 0),
            "total_raw_score": learned_region_scored.get("total_raw_score", 0),
            "task_rule": learned_region_scored.get("task_rule"),
        }

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

    best_task_rule = None

    if best_strategy == "learned_region_rule":
        best_task_rule = strategy_stats["learned_region_rule"].get("task_rule")

    return {
        "best_strategy": best_strategy,
        "strategy_stats": strategy_stats,
        "task_rule": best_task_rule,
        "rule": best_task_rule,
    }


# ============================================================
# FORCED PAIR-LEVEL STRATEGY SOLVER
# ============================================================

def solve_pair_with_forced_strategy(input_grid, output_grid, strategy_name):
    """
    Force a normal pair-level strategy.

    Note:
    task-level strategies require a learned task_rule, so use
    apply_task_rule_to_input(...) for them.
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

    elif strategy_name == "learned_region_rule":
        print(
            "[FORCED STRATEGY ERROR] learned_region_rule needs a learned "
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

    Some strategies are true task-level rules and can run on test inputs
    without expected output.

    Other strategies are still pair-level discovery rules. Those need
    expected_grid during debugging/training.
    """

    # --------------------------------------------------------
    # Multi-seed task-level rule
    # --------------------------------------------------------
    if strategy_name == "multi_seed_composition_rule":
        if task_rule is None:
            print("[TASK RULE ERROR] Missing multi_seed task_rule.")
            return None

        # Train replay.
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

        # Test application.
        if apply_multi_seed_composition_rule is None:
            print("[TASK RULE ERROR] apply_multi_seed_composition_rule import failed.")
            return None

        return apply_multi_seed_composition_rule(
            task_rule,
            input_grid,
        )

    # --------------------------------------------------------
    # Learned region task-level rule
    # --------------------------------------------------------
    if strategy_name == "learned_region_rule":
        if task_rule is None:
            print("[TASK RULE ERROR] Missing learned_region task_rule.")
            return None

        if apply_learned_region_rule is None:
            print("[TASK RULE ERROR] apply_learned_region_rule import failed.")
            return None

        return apply_learned_region_rule(
            task_rule,
            input_grid,
        )

    # --------------------------------------------------------
    # Normal pair-level strategies
    # --------------------------------------------------------
    if expected_grid is None:
        print(
            f"[TEST APPLY SKIPPED] {strategy_name} has no learned task_rule yet, "
            "and expected_grid is None."
        )
        return None

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