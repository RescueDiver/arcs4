# reasoning/task_router.py

# ============================================================
# NORMAL PAIR-LEVEL STRATEGIES
# ============================================================

from reasoning.pattern_rule_engine import solve_pair_pattern_rule
from reasoning.region_rule_engine import solve_pair_region_rule
from reasoning.region_alignment_rule_engine_v2 import solve_pair_region_alignment_rule_v2
from reasoning.motif_layout_rule import solve_pair_motif_layout_rule
from reasoning.object_grid_rule import solve_pair_object_grid_rule
from reasoning.pattern_expansion_rule import solve_pair_pattern_expansion
from reasoning.seed_placement_expansion_rule import solve_pair_seed_placement_expansion
from reasoning.anchor_compass_merge_rule import predict_anchor_compass_merge_for_pair

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
# OPTIONAL TASK-LEVEL RING/BLOB RULE SYNTHESIZER
# ============================================================

try:
    from reasoning.archive.ring_blob_rule_synthesizer import (
        learn_ring_blob_rule_synthesizer,
        predict_with_ring_blob_rule_synthesizer,
    )
except ImportError:
    learn_ring_blob_rule_synthesizer = None
    predict_with_ring_blob_rule_synthesizer = None


# ============================================================
# OPTIONAL TASK-LEVEL VISUAL SYMBOLIC RULE
# ============================================================

try:
    from reasoning.visual_symbolic_rule import (
        discover_visual_symbolic_rule_for_task,
        apply_visual_symbolic_rule,
        score_visual_symbolic_rule_on_train,
        leave_one_out_visual_symbolic,
    )
except ImportError as exc:
    print("[VISUAL SYMBOLIC IMPORT ERROR]", repr(exc))

    discover_visual_symbolic_rule_for_task = None
    apply_visual_symbolic_rule = None
    score_visual_symbolic_rule_on_train = None
    leave_one_out_visual_symbolic = None


try:
    from reasoning.visual_symbolic_ruleV2 import (
        discover_visual_symbolic_rule_v2_for_task,
        apply_visual_symbolic_rule_v2,
        extract_scene_facts,
    )
except ImportError as exc:
    print("[VISUAL SYMBOLIC V2 IMPORT ERROR]", repr(exc))

    discover_visual_symbolic_rule_v2_for_task = None
    apply_visual_symbolic_rule_v2 = None
    extract_scene_facts = None


# ============================================================
# OPTIONAL TASK-LEVEL VISUAL SYMBOLIC RULE V2
# ============================================================

try:
    from reasoning.visual_symbolic_ruleV2 import (
        explain_visual_symbolic_prediction,
        extract_scene_facts,
    )

    from reasoning.visual_symbolic_output_learner import (
        build_marker_learning_example,
        learn_marker_placement_rule,
        apply_learned_marker_rule,
    )

except ImportError as exc:
    print("[VISUAL SYMBOLIC V2 IMPORT ERROR]", repr(exc))

    explain_visual_symbolic_prediction = None
    extract_scene_facts = None
    build_marker_learning_example = None
    learn_marker_placement_rule = None
    apply_learned_marker_rule = None


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
    Normalize one pair-level strategy result into the router's candidate format.
    """
    if result is None:
        return None

    pred = result.get("predicted")

    if pred is None:
        pred = result.get("prediction")

    if pred is None:
        return None

    result["predicted"] = pred

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

        for placement in placements:
            if placement.get("ratio", 0) < 1.0:
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
    if learned_rule is None or apply_learned_region_rule is None:
        return None

    total_score = 0
    exact_count = 0
    pair_count = 0
    results = []

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        output_grid = pair["output"]

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
    if scored_rule is None:
        return False

    pair_count = scored_rule.get("pair_count", 0)
    exact_count = scored_rule.get("exact_count", 0)
    results = scored_rule.get("results", [])

    if pair_count == 0:
        return False

    if len(results) != pair_count:
        return False

    if exact_count != pair_count:
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
# RING/BLOB SYNTHESIZER HELPERS
# ============================================================

def score_ring_blob_rule_synthesizer_on_train(task_rule, train_pairs):
    if task_rule is None:
        return None

    if predict_with_ring_blob_rule_synthesizer is None:
        return None

    total_score = 0
    exact_count = 0
    pair_count = 0
    results = []

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        output_grid = pair["output"]

        try:
            result = predict_with_ring_blob_rule_synthesizer(
                task_rule,
                input_grid,
            )
            predicted = result.get("prediction")
            details = result

        except Exception as exc:
            predicted = None
            details = {
                "error": repr(exc),
            }

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
            "details": details,
        })

    return {
        "strategy": "ring_blob_rule_synthesizer",
        "task_rule": task_rule,
        "rule": task_rule,
        "pair_count": pair_count,
        "exact_count": exact_count,
        "total_raw_score": total_score,
        "total_adjusted_score": total_score,
        "results": results,
    }


def is_strong_ring_blob_rule_synthesizer_result(scored_rule):
    if scored_rule is None:
        return False

    pair_count = scored_rule.get("pair_count", 0)
    exact_count = scored_rule.get("exact_count", 0)
    results = scored_rule.get("results", [])

    if pair_count == 0:
        return False

    if len(results) != pair_count:
        return False

    if exact_count != pair_count:
        return False

    for item in results:
        if item.get("predicted") is None:
            return False

    return True


def build_ring_blob_strategy_stats(scored_rule):
    return {
        "ring_blob_rule_synthesizer": {
            "pair_count": scored_rule.get("pair_count", 0),
            "exact_count": scored_rule.get("exact_count", 0),
            "total_adjusted_score": scored_rule.get("total_adjusted_score", 0),
            "total_raw_score": scored_rule.get("total_raw_score", 0),
            "task_rule": scored_rule.get("task_rule"),
        }
    }


# ============================================================
# VISUAL SYMBOLIC RULE HELPERS
# ============================================================

def build_visual_symbolic_strategy_stats(scored_rule, loo_scored=None):
    return {
        "visual_symbolic_rule": {
            "pair_count": scored_rule.get("pair_count", 0) if scored_rule else 0,
            "exact_count": scored_rule.get("exact_count", 0) if scored_rule else 0,
            "total_adjusted_score": scored_rule.get("total_adjusted_score", 0) if scored_rule else 0,
            "total_raw_score": scored_rule.get("total_raw_score", 0) if scored_rule else 0,
            "loo_pair_count": loo_scored.get("pair_count", 0) if loo_scored else 0,
            "loo_exact_count": loo_scored.get("exact_count", 0) if loo_scored else 0,
            "loo_total_score": loo_scored.get("total_raw_score", 0) if loo_scored else 0,
        }
    }


def is_strong_visual_symbolic_leave_one_out(loo_scored):
    """
    Honest gate for visual_symbolic_rule.

    It can override only if it predicts every hidden train pair exactly.
    """
    if loo_scored is None:
        return False

    pair_count = loo_scored.get("pair_count", 0)
    exact_count = loo_scored.get("exact_count", 0)
    results = loo_scored.get("results", [])

    if pair_count == 0:
        return False

    if len(results) != pair_count:
        return False

    if exact_count != pair_count:
        return False

    for item in results:
        if item.get("predicted") is None:
            return False

    return True


def is_ring_blob_style_task(train_pairs):
    """
    Gate visual_symbolic_ruleV2.

    This prevents the ring/blob solver from running on unrelated ARC tasks.
    It only checks whether the task has rings + blobs.
    """

    if extract_scene_facts is None:
        return False

    if not train_pairs:
        return False

    for pair in train_pairs:
        input_grid = pair.get("input")

        if input_grid is None:
            return False

        try:
            scene = extract_scene_facts(input_grid)
        except Exception:
            return False

        ring_count = scene.get("ring_count", 0)
        blob_count = scene.get("blob_count", 0)

        if ring_count < 1:
            return False

        if blob_count < 1:
            return False

    return True


def discover_visual_symbolic_rule_v2_for_router(train_pairs):
    """
    Learn the real V2 task-level rule:
        frame prediction + learned marker placement.

    This matches the successful run_onceV3 path.
    """

    if explain_visual_symbolic_prediction is None:
        return None

    if build_marker_learning_example is None:
        return None

    if learn_marker_placement_rule is None:
        return None

    learning_examples = []

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        output_grid = pair["output"]

        explanation = explain_visual_symbolic_prediction(input_grid)

        learning_example = build_marker_learning_example(
            pair_index=pair_index,
            input_grid=input_grid,
            output_grid=output_grid,
            explanation=explanation,
        )

        learning_examples.append(learning_example)

    marker_rule = learn_marker_placement_rule(learning_examples)

    return {
        "family": "visual_symbolic_ruleV2",
        "mode": "learned_marker_router_rule",
        "marker_rule": marker_rule,
    }


def apply_visual_symbolic_rule_v2_for_router(task_rule, input_grid):
    """
    Apply the learned V2 marker rule.

    This is not the frame-only result.
    It does:
        1. explain/build frame prediction
        2. apply learned output markers
    """

    if task_rule is None:
        return None

    if explain_visual_symbolic_prediction is None:
        return None

    if apply_learned_marker_rule is None:
        return None

    marker_rule = task_rule.get("marker_rule")

    if marker_rule is None:
        return None

    explanation = explain_visual_symbolic_prediction(input_grid)
    frame_prediction = explanation.get("prediction")

    if frame_prediction is None:
        return None

    learned_prediction, applied_markers = apply_learned_marker_rule(
        prediction=frame_prediction,
        explanation=explanation,
        marker_rule=marker_rule,
    )

    return learned_prediction


def score_visual_symbolic_v2_on_train(task_rule, train_pairs):
    if task_rule is None:
        return None

    total_score = 0
    exact_count = 0
    pair_count = 0
    results = []

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        output_grid = pair["output"]

        try:
            predicted = apply_visual_symbolic_rule_v2_for_router(
                task_rule,
                input_grid,
            )

        except Exception as exc:
            predicted = None
            print(
                "[VISUAL SYMBOLIC V2 SCORE WARNING]",
                f"pair={pair_index}",
                repr(exc),
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
        "strategy": "visual_symbolic_ruleV2",
        "task_rule": task_rule,
        "rule": task_rule,
        "pair_count": pair_count,
        "exact_count": exact_count,
        "total_raw_score": total_score,
        "total_adjusted_score": total_score,
        "results": results,
    }


def leave_one_out_visual_symbolic_v2(train_pairs):
    """
    Diagnostic only.

    We do not require perfect LOO for this family because some hidden pairs
    remove the only example of a needed marker behavior.
    """

    total_score = 0
    exact_count = 0
    pair_count = 0
    results = []

    for hidden_index in range(len(train_pairs)):
        visible_pairs = [
            pair
            for idx, pair in enumerate(train_pairs)
            if idx != hidden_index
        ]

        hidden_pair = train_pairs[hidden_index]

        try:
            rule = discover_visual_symbolic_rule_v2_for_router(
                visible_pairs,
            )

            predicted = apply_visual_symbolic_rule_v2_for_router(
                rule,
                hidden_pair["input"],
            )

        except Exception as exc:
            predicted = None
            print(
                "[VISUAL SYMBOLIC V2 LOO WARNING]",
                f"hidden_pair={hidden_index}",
                repr(exc),
            )

        expected = hidden_pair["output"]

        score = score_prediction(predicted, expected)
        exact = predicted == expected

        if exact:
            exact_count += 1

        pair_count += 1
        total_score += score

        results.append({
            "pair_index": hidden_index,
            "predicted": predicted,
            "score": score,
            "exact": exact,
        })

    return {
        "strategy": "visual_symbolic_ruleV2",
        "pair_count": pair_count,
        "exact_count": exact_count,
        "total_raw_score": total_score,
        "total_adjusted_score": total_score,
        "results": results,
    }


def is_strong_visual_symbolic_v2_result(scored_rule, loo_scored):
    """
    Gate for allowing visual_symbolic_ruleV2 to control the task.

    Requirements:
        1. It must solve every visible train pair.
        2. It must have at least some honest LOO success.
        3. It must only run on ring/blob-style tasks.
    """

    if scored_rule is None:
        return False

    pair_count = scored_rule.get("pair_count", 0)
    exact_count = scored_rule.get("exact_count", 0)

    if pair_count == 0:
        return False

    if exact_count != pair_count:
        return False

    if loo_scored is None:
        return False

    loo_pair_count = loo_scored.get("pair_count", 0)
    loo_exact_count = loo_scored.get("exact_count", 0)

    if loo_pair_count == 0:
        return False

    # 50% LOO is acceptable here because hidden pairs may remove unique evidence.
    if loo_exact_count < max(1, loo_pair_count // 2):
        return False

    return True


def build_visual_symbolic_v2_strategy_stats(scored_rule, loo_scored=None):
    return {
        "visual_symbolic_ruleV2": {
            "pair_count": scored_rule.get("pair_count", 0) if scored_rule else 0,
            "exact_count": scored_rule.get("exact_count", 0) if scored_rule else 0,
            "total_adjusted_score": scored_rule.get("total_adjusted_score", 0) if scored_rule else 0,
            "total_raw_score": scored_rule.get("total_raw_score", 0) if scored_rule else 0,
            "loo_pair_count": loo_scored.get("pair_count", 0) if loo_scored else 0,
            "loo_exact_count": loo_scored.get("exact_count", 0) if loo_scored else 0,
            "loo_total_score": loo_scored.get("total_raw_score", 0) if loo_scored else 0,
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
            f" {result.get('strategy'):<34} "
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
            print(f"{name:<34}: None")
        else:
            print(f"{name:<34}: {result.get('score')}")


def debug_router_adjustments(result_seed_placement, result_pattern, result_region, result_motif_layout, result_region_alignment_v2,
         result_object_grid, result_pattern_expansion,):

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
            print(f"{name:<34}: None")
            continue

        pred = result.get("predicted")
        ph, pw = grid_shape(pred)

        print(
            f"{name:<34} "
            f"raw={result.get('raw_score'):<7} "
            f"adj={result.get('adjusted_score'):<7} "
            f"shape_pen={result.get('shape_penalty'):<4} "
            f"full_pen={result.get('full_grid_penalty'):<4} "
            f"out={ph}x{pw}"
        )


def print_task_level_replay_debug(strategy_name, scored_rule):
    print()
    print(f"[TASK-LEVEL DEBUG] {strategy_name}")
    print("-" * 60)

    if scored_rule is None:
        print("No scored rule.")
        return

    print(f"Exact count: {scored_rule.get('exact_count')}")
    print(f"Pair count : {scored_rule.get('pair_count')}")
    print(f"Total score: {scored_rule.get('total_raw_score')}")

    for item in scored_rule.get("results", []):
        pred = item.get("predicted")
        ph, pw = grid_shape(pred)

        print(
            f"  pair {item.get('pair_index')}: "
            f"exact={item.get('exact')} "
            f"score={item.get('score')} "
            f"shape={ph}x{pw}"
        )


def print_visual_symbolic_debug(visual_symbolic_scored, visual_symbolic_loo):
    print()
    print("[TASK-LEVEL DEBUG] visual_symbolic_rule")
    print("-" * 60)

    if visual_symbolic_scored is None:
        print("Train replay: None")
    else:
        print("Train replay:")
        print(f"  exact_count: {visual_symbolic_scored.get('exact_count')}")
        print(f"  pair_count : {visual_symbolic_scored.get('pair_count')}")
        print(f"  total_score: {visual_symbolic_scored.get('total_raw_score')}")

        for item in visual_symbolic_scored.get("results", []):
            pred = item.get("predicted")
            ph, pw = grid_shape(pred)

            print(
                f"  pair {item.get('pair_index')}: "
                f"exact={item.get('exact')} "
                f"score={item.get('score')} "
                f"shape={ph}x{pw}"
            )

    print()

    if visual_symbolic_loo is None:
        print("Leave-one-out: None")
    else:
        print("Leave-one-out:")
        print(f"  exact_count: {visual_symbolic_loo.get('exact_count')}")
        print(f"  pair_count : {visual_symbolic_loo.get('pair_count')}")
        print(f"  total_score: {visual_symbolic_loo.get('total_raw_score')}")

        for item in visual_symbolic_loo.get("results", []):
            pred = item.get("predicted")
            ph, pw = grid_shape(pred)

            print(
                f"  hidden pair {item.get('pair_index')}: "
                f"exact={item.get('exact')} "
                f"score={item.get('score')} "
                f"shape={ph}x{pw}"
            )


# ============================================================
# ACTIVE PAIR ROUTER
# ============================================================

def get_all_strategy_results(input_grid, output_grid, debug=True):
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

    if task_type in ["pattern_same_size", "general", "motif_layout", "expansion"]:
        result_pattern = maybe_add_candidate(
            candidates,
            solve_pair_pattern_rule(input_grid, output_grid),
            "pattern_rule",
            input_grid,
            output_grid,
        )

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

    if task_type == "expansion":
        result_pattern_expansion = maybe_add_candidate(
            candidates,
            solve_pair_pattern_expansion(input_grid, output_grid),
            "pattern_expansion_rule",
            input_grid,
            output_grid,
        )

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
        key=lambda result: (
            1 if result.get("exact") else 0,
            result.get("adjusted_score", result.get("score", -10**9)),
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

    Honest rule:
        Task-level rules may only override if they pass their strict gate.

    Current gates:
        multi_seed_composition_rule:
            strict internal seed match gate

        visual_symbolic_rule:
            honest leave-one-out gate

        learned_region_rule:
            blocked for now because it replayed visible train pairs

        ring_blob_rule_synthesizer:
            blocked for now unless later given leave-one-out gate
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
    # 2. RING/BLOB SYNTHESIZER — DEBUG ONLY FOR NOW
    # --------------------------------------------------------
    ring_blob_rule = None
    ring_blob_scored = None

    if learn_ring_blob_rule_synthesizer is not None:
        try:
            ring_blob_rule = learn_ring_blob_rule_synthesizer(train_pairs)

            ring_blob_scored = score_ring_blob_rule_synthesizer_on_train(
                ring_blob_rule,
                train_pairs,
            )

            if debug:
                print_task_level_replay_debug(
                    "ring_blob_rule_synthesizer",
                    ring_blob_scored,
                )

            if is_strong_ring_blob_rule_synthesizer_result(ring_blob_scored):
                if debug:
                    print("\n[RING/BLOB BLOCKED]")
                    print("ring_blob_rule_synthesizer solved visible train pairs.")
                    print("Blocked because visible-pair replay is not honest learning.")
                    print("It must pass leave-one-out before it can override.")

        except Exception as exc:
            if debug:
                print("[RING/BLOB WARNING] ring_blob_rule_synthesizer failed")
                print("  error:", repr(exc))
    else:
        if debug:
            print("[RING/BLOB WARNING] ring_blob_rule_synthesizer import failed")

    # --------------------------------------------------------
    # 3. VISUAL SYMBOLIC RULE V2 — RING/BLOB TASK-LEVEL RULE
    # --------------------------------------------------------
    visual_symbolic_v2_rule = None
    visual_symbolic_v2_scored = None
    visual_symbolic_v2_loo = None

    if is_ring_blob_style_task(train_pairs):
        if discover_visual_symbolic_rule_v2_for_router is not None:
            try:
                visual_symbolic_v2_rule = discover_visual_symbolic_rule_v2_for_router(
                    train_pairs,
                )

                visual_symbolic_v2_scored = score_visual_symbolic_v2_on_train(
                    visual_symbolic_v2_rule,
                    train_pairs,
                )

                visual_symbolic_v2_loo = leave_one_out_visual_symbolic_v2(
                    train_pairs,
                )

                if debug:
                    print_task_level_replay_debug(
                        "visual_symbolic_ruleV2",
                        visual_symbolic_v2_scored,
                    )

                    print_task_level_replay_debug(
                        "visual_symbolic_ruleV2 leave-one-out",
                        visual_symbolic_v2_loo,
                    )

                if is_strong_visual_symbolic_v2_result(
                    visual_symbolic_v2_scored,
                    visual_symbolic_v2_loo,
                ):
                    if debug:
                        print("\n[ROUTER OVERRIDE] Using visual_symbolic_ruleV2")
                        print("[ROUTER OVERRIDE] Reason: ring/blob task, full train exact, LOO acceptable")

                    stats = build_visual_symbolic_v2_strategy_stats(
                        visual_symbolic_v2_scored,
                        visual_symbolic_v2_loo,
                    )

                    return {
                        "best_strategy": "visual_symbolic_ruleV2",
                        "strategy_stats": stats,
                        "task_rule": visual_symbolic_v2_rule,
                        "rule": visual_symbolic_v2_rule,
                    }

                else:
                    if debug:
                        print("\n[VISUAL SYMBOLIC V2 BLOCKED]")
                        print("visual_symbolic_ruleV2 did not pass its task-level gate.")

            except Exception as exc:
                if debug:
                    print("[VISUAL SYMBOLIC V2 WARNING] visual_symbolic_ruleV2 failed")
                    print("  error:", repr(exc))
        else:
            if debug:
                print("[VISUAL SYMBOLIC V2 WARNING] visual_symbolic_ruleV2 import failed")
    else:
        if debug:
            print("[VISUAL SYMBOLIC V2 SKIPPED] not a ring/blob-style task")


    # --------------------------------------------------------
    # 3. LEARNED REGION RULE — DEBUG ONLY FOR NOW
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

                print_task_level_replay_debug(
                    "learned_region_rule",
                    learned_region_scored,
                )

            if is_strong_learned_region_result(learned_region_scored):
                if debug:
                    print("\n[LEARNED REGION BLOCKED]")
                    print("learned_region_rule solved visible train pairs.")
                    print("Blocked because visible-pair replay is not honest learning.")
                    print("It must pass leave-one-out before it can override.")
    else:
        if debug:
            print("[LEARNED REGION WARNING] learned_region_rule import failed")

    # --------------------------------------------------------
    # 4. OLD VISUAL SYMBOLIC RULE — DISABLED
    # --------------------------------------------------------
    visual_symbolic_rule = None
    visual_symbolic_scored = None
    visual_symbolic_loo = None

    if debug:
        print("[OLD VISUAL SYMBOLIC SKIPPED] disabled; use visual_symbolic_ruleV2 only")

    # --------------------------------------------------------
    # 5. NORMAL PAIR-LEVEL STRATEGY RANKING
    # --------------------------------------------------------
    strategy_stats = {}

    if ring_blob_scored is not None:
        strategy_stats["BLOCKED_ring_blob_rule_synthesizer"] = {
            "pair_count": ring_blob_scored.get("pair_count", 0),
            "exact_count": ring_blob_scored.get("exact_count", 0),
            "total_adjusted_score": ring_blob_scored.get("total_adjusted_score", 0),
            "total_raw_score": ring_blob_scored.get("total_raw_score", 0),
            "task_rule": ring_blob_scored.get("task_rule"),
            "blocked": True,
        }

    if learned_region_scored is not None:
        strategy_stats["BLOCKED_learned_region_rule"] = {
            "pair_count": learned_region_scored.get("pair_count", 0),
            "exact_count": learned_region_scored.get("exact_count", 0),
            "total_adjusted_score": learned_region_scored.get("total_adjusted_score", 0),
            "total_raw_score": learned_region_scored.get("total_raw_score", 0),
            "task_rule": learned_region_scored.get("task_rule"),
            "blocked": True,
        }

    if visual_symbolic_scored is not None:
        strategy_stats["BLOCKED_visual_symbolic_rule"] = {
            "pair_count": visual_symbolic_scored.get("pair_count", 0),
            "exact_count": visual_symbolic_scored.get("exact_count", 0),
            "total_adjusted_score": visual_symbolic_scored.get("total_adjusted_score", 0),
            "total_raw_score": visual_symbolic_scored.get("total_raw_score", 0),
            "loo_pair_count": visual_symbolic_loo.get("pair_count", 0) if visual_symbolic_loo else 0,
            "loo_exact_count": visual_symbolic_loo.get("exact_count", 0) if visual_symbolic_loo else 0,
            "task_rule": visual_symbolic_scored.get("task_rule"),
            "blocked": True,
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
                    "blocked": False,
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
        if stats.get("blocked"):
            continue

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

    if strategy_name == "pattern_rule":
        result = solve_pair_pattern_rule(input_grid, output_grid)

    elif strategy_name == "region_rule":
        result = solve_pair_region_rule(input_grid, output_grid)

    elif strategy_name == "visual_symbolic_ruleV2":
        print(
            "[FORCED STRATEGY ERROR] visual_symbolic_ruleV2 needs a learned "
            "task_rule. Use apply_task_rule_to_input(...)."
        )
        return None

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

    elif strategy_name == "ring_blob_rule_synthesizer":
        print(
            "[FORCED STRATEGY ERROR] ring_blob_rule_synthesizer needs a learned "
            "task_rule. Use apply_task_rule_to_input(...)."
        )
        return None

    elif strategy_name == "learned_region_rule":
        print(
            "[FORCED STRATEGY ERROR] learned_region_rule needs a learned "
            "task_rule. Use apply_task_rule_to_input(...)."
        )
        return None

    elif strategy_name == "visual_symbolic_rule":
        print(
            "[FORCED STRATEGY ERROR] visual_symbolic_rule needs a learned "
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

def apply_task_rule_to_input(strategy_name, task_rule, input_grid, expected_grid=None, pair_index=None,):
    # --------------------------------------------------------
    # Multi-seed task-level rule
    # --------------------------------------------------------
    if strategy_name == "multi_seed_composition_rule":
        if task_rule is None:
            print("[TASK RULE ERROR] Missing multi_seed task_rule.")
            return None

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

        if apply_multi_seed_composition_rule is None:
            print("[TASK RULE ERROR] apply_multi_seed_composition_rule import failed.")
            return None

        return apply_multi_seed_composition_rule(
            task_rule,
            input_grid,
        )

    # --------------------------------------------------------
    # Ring/blob task-level rule
    # --------------------------------------------------------
    if strategy_name == "ring_blob_rule_synthesizer":
        if task_rule is None:
            print("[TASK RULE ERROR] Missing ring_blob_rule_synthesizer task_rule.")
            return None

        if predict_with_ring_blob_rule_synthesizer is None:
            print("[TASK RULE ERROR] predict_with_ring_blob_rule_synthesizer import failed.")
            return None

        result = predict_with_ring_blob_rule_synthesizer(
            task_rule,
            input_grid,
        )

        if result is None:
            return None

        return result.get("prediction")

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
    # Visual symbolic task-level rule
    # --------------------------------------------------------
    if strategy_name == "visual_symbolic_rule":
        if task_rule is None:
            print("[TASK RULE ERROR] Missing visual_symbolic_rule task_rule.")
            return None

        if apply_visual_symbolic_rule is None:
            print("[TASK RULE ERROR] apply_visual_symbolic_rule import failed.")
            return None

        return apply_visual_symbolic_rule(
            task_rule,
            input_grid,
        )

    # --------------------------------------------------------
    # Visual symbolic V2 task-level rule
    # --------------------------------------------------------
    if strategy_name == "visual_symbolic_ruleV2":
        if task_rule is None:
            print("[TASK RULE ERROR] Missing visual_symbolic_ruleV2 task_rule.")
            return None

        return apply_visual_symbolic_rule_v2_for_router(
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


# ============================================================
# TEST-ONLY TASK CONTEXT HELPERS
# ============================================================

def try_anchor_compass_merge_for_test_pair(
    task,
    test_pair,
    test_index=None,
    debug=False,
):
    """
    Test-only helper.

    This rule needs the full task because it learns templates from train pairs,
    then applies them to a test pair.

    It returns a normal router-style result if it fires.
    It returns None if this test pair should fall back.
    """

    prediction = predict_anchor_compass_merge_for_pair(
        task=task,
        pair=test_pair,
        test_index=test_index,
        debug=debug,
    )

    if prediction is None:
        return None

    return {
        "strategy": "anchor_compass_merge_rule",
        "predicted": prediction,
        "prediction": prediction,
        "score": 0,
        "adjusted_score": 0,
        "exact": False,
        "task_rule": None,
        "rule": None,
    }



def score_task_rule_prediction(strategy_name, task_rule, input_grid, expected_grid, pair_index=None,):
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
