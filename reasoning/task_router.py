# reasoning/task_router.py
"""
Clean 5-family task router for ARCs4.

Important honesty rule:
    One task = one family = one fixed inner strategy.

The router may test several candidate helpers during training,
but once a family chooses one helper, every train pair and test pair
must use that same helper.

This removes fake 100% train scores caused by picking a different
best helper for each train pair.
"""

from copy import deepcopy


# ============================================================
# SAFE IMPORT HELPERS
# ============================================================

def _optional_import(import_fn):
    try:
        return import_fn()
    except Exception:
        return None


def _safe_call(fn, *args, **kwargs):
    if fn is None:
        return None

    try:
        return fn(*args, **kwargs)
    except TypeError:
        try:
            return fn(*args)
        except Exception:
            return None
    except Exception:
        return None


# ============================================================
# OPTIONAL OLD PAIR-LEVEL ENGINES
# ============================================================

solve_pair_pattern_rule = _optional_import(
    lambda: __import__(
        "reasoning.pattern_rule_engine",
        fromlist=["solve_pair_pattern_rule"],
    ).solve_pair_pattern_rule
)

solve_pair_region_rule = _optional_import(
    lambda: __import__(
        "reasoning.region_rule_engine",
        fromlist=["solve_pair_region_rule"],
    ).solve_pair_region_rule
)

solve_pair_region_alignment_rule_v2 = _optional_import(
    lambda: __import__(
        "reasoning.region_alignment_rule_engine_v2",
        fromlist=["solve_pair_region_alignment_rule_v2"],
    ).solve_pair_region_alignment_rule_v2
)

solve_pair_motif_layout_rule = _optional_import(
    lambda: __import__(
        "reasoning.motif_layout_rule",
        fromlist=["solve_pair_motif_layout_rule"],
    ).solve_pair_motif_layout_rule
)

solve_pair_object_grid_rule = _optional_import(
    lambda: __import__(
        "reasoning.object_grid_rule",
        fromlist=["solve_pair_object_grid_rule"],
    ).solve_pair_object_grid_rule
)

solve_pair_pattern_expansion = _optional_import(
    lambda: __import__(
        "reasoning.pattern_expansion_rule",
        fromlist=["solve_pair_pattern_expansion"],
    ).solve_pair_pattern_expansion
)

solve_pair_seed_placement_expansion = _optional_import(
    lambda: __import__(
        "reasoning.seed_placement_expansion_rule",
        fromlist=["solve_pair_seed_placement_expansion"],
    ).solve_pair_seed_placement_expansion
)

predict_anchor_compass_merge_for_pair = _optional_import(
    lambda: __import__(
        "reasoning.anchor_compass_merge_rule",
        fromlist=["predict_anchor_compass_merge_for_pair"],
    ).predict_anchor_compass_merge_for_pair
)


# ============================================================
# OPTIONAL TASK-LEVEL ENGINES
# ============================================================

discover_multi_seed_composition_rule_for_task = _optional_import(
    lambda: __import__(
        "reasoning.multi_seed_composition_rule",
        fromlist=["discover_multi_seed_composition_rule_for_task"],
    ).discover_multi_seed_composition_rule_for_task
)

apply_multi_seed_composition_rule = _optional_import(
    lambda: __import__(
        "reasoning.multi_seed_composition_rule",
        fromlist=["apply_multi_seed_composition_rule"],
    ).apply_multi_seed_composition_rule
)

apply_multi_seed_composition_rule_for_train_pair = _optional_import(
    lambda: __import__(
        "reasoning.multi_seed_composition_rule",
        fromlist=["apply_multi_seed_composition_rule_for_train_pair"],
    ).apply_multi_seed_composition_rule_for_train_pair
)

discover_learned_region_rule_for_task = _optional_import(
    lambda: __import__(
        "reasoning.learned_region_rule",
        fromlist=["discover_learned_region_rule_for_task"],
    ).discover_learned_region_rule_for_task
)

apply_learned_region_rule = _optional_import(
    lambda: __import__(
        "reasoning.learned_region_rule",
        fromlist=["apply_learned_region_rule"],
    ).apply_learned_region_rule
)

discover_visual_symbolic_rule_for_task = _optional_import(
    lambda: __import__(
        "reasoning.visual_symbolic_rule",
        fromlist=["discover_visual_symbolic_rule_for_task"],
    ).discover_visual_symbolic_rule_for_task
)

apply_visual_symbolic_rule = _optional_import(
    lambda: __import__(
        "reasoning.visual_symbolic_rule",
        fromlist=["apply_visual_symbolic_rule"],
    ).apply_visual_symbolic_rule
)

discover_anchor_repair_rule_for_task = _optional_import(
    lambda: __import__(
        "reasoning.anchor_repair_rule",
        fromlist=["discover_anchor_repair_rule_for_task"],
    ).discover_anchor_repair_rule_for_task
)

apply_anchor_repair_rule = _optional_import(
    lambda: __import__(
        "reasoning.anchor_repair_rule",
        fromlist=["apply_anchor_repair_rule"],
    ).apply_anchor_repair_rule
)


# ============================================================
# BASIC GRID HELPERS
# ============================================================

def grid_shape(grid):
    if grid is None:
        return 0, 0

    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def copy_grid(grid):
    if grid is None:
        return None
    return deepcopy(grid)


def is_valid_arc_grid(grid):
    """Return True only for a rectangular ARC grid accepted by Kaggle."""
    if not isinstance(grid, list) or not 1 <= len(grid) <= 30:
        return False

    if not isinstance(grid[0], list) or not 1 <= len(grid[0]) <= 30:
        return False

    width = len(grid[0])

    for row in grid:
        if not isinstance(row, list) or len(row) != width:
            return False

        for value in row:
            if isinstance(value, bool) or not isinstance(value, int):
                return False
            if not 0 <= value <= 9:
                return False

    return True


def score_prediction(predicted, expected):
    """
    Larger is better.
    Exact match gets a huge bonus.
    """
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
        return 10_000

    ph, pw = grid_shape(predicted)
    eh, ew = grid_shape(expected)

    if ph == eh and pw == ew:
        return 0

    return (abs(ph - eh) + abs(pw - ew)) * 50


def extract_prediction(result):
    if result is None:
        return None

    if isinstance(result, dict):
        pred = result.get("predicted")
        if pred is None:
            pred = result.get("prediction")
        return pred

    if isinstance(result, list):
        return result

    return None


def normalize_prediction_result(
    strategy,
    predicted,
    expected=None,
    raw_result=None,
):
    if predicted is None:
        return None

    raw_result = dict(raw_result or {})

    raw_score = score_prediction(predicted, expected) if expected is not None else 0
    penalty = shape_penalty(predicted, expected) if expected is not None else 0
    adjusted_score = raw_score - penalty

    raw_result["strategy"] = strategy
    raw_result["predicted"] = predicted
    raw_result["prediction"] = predicted
    raw_result["score"] = raw_score
    raw_result["raw_score"] = raw_score
    raw_result["adjusted_score"] = adjusted_score
    raw_result["total_adjusted_score"] = adjusted_score
    raw_result["shape_penalty"] = penalty
    raw_result["exact"] = predicted == expected if expected is not None else False

    return raw_result


# ============================================================
# TASK FEATURE HINTS
# ============================================================

def find_divider_column(grid):
    if grid is None:
        return None

    h, w = grid_shape(grid)
    if h == 0 or w == 0:
        return None

    for c in range(w):
        values = [grid[r][c] for r in range(h)]

        if len(set(values)) == 1 and values[0] != 0:
            return c

    return None


def detect_task_features(train_pairs):
    features = {
        "pair_count": len(train_pairs),
        "same_size_count": 0,
        "crop_like_count": 0,
        "expansion_count": 0,
        "has_divider_count": 0,
        "many_color_count": 0,
    }

    for pair in train_pairs:
        inp = pair.get("input")
        out = pair.get("output")

        ih, iw = grid_shape(inp)
        oh, ow = grid_shape(out)

        if ih == oh and iw == ow:
            features["same_size_count"] += 1

        if oh <= ih and ow <= iw and (oh, ow) != (ih, iw):
            features["crop_like_count"] += 1

        if oh > ih or ow > iw:
            features["expansion_count"] += 1

        if find_divider_column(inp) is not None:
            features["has_divider_count"] += 1

        colors = set()
        if inp is not None:
            for row in inp:
                colors.update(row)

        if len(colors) >= 4:
            features["many_color_count"] += 1

    return features


# ============================================================
# LEGACY STRATEGY TABLE
# ============================================================

LEGACY_PAIR_STRATEGIES = {
    "pattern_rule": solve_pair_pattern_rule,
    "region_rule": solve_pair_region_rule,
    "region_alignment_rule_v2": solve_pair_region_alignment_rule_v2,
    "motif_layout_rule": solve_pair_motif_layout_rule,
    "object_grid_rule": solve_pair_object_grid_rule,
    "pattern_expansion_rule": solve_pair_pattern_expansion,
    "seed_placement_expansion_rule": solve_pair_seed_placement_expansion,
}


def run_legacy_pair_strategy(strategy_name, input_grid, output_grid=None):
    fn = LEGACY_PAIR_STRATEGIES.get(strategy_name)

    if fn is None:
        return None

    result = _safe_call(fn, input_grid, output_grid)
    predicted = extract_prediction(result)

    return normalize_prediction_result(
        strategy=strategy_name,
        predicted=predicted,
        expected=output_grid,
        raw_result=result if isinstance(result, dict) else None,
    )


def solve_pair_with_forced_strategy(input_grid, output_grid, strategy_name):
    return run_legacy_pair_strategy(strategy_name, input_grid, output_grid)


def solve_pair_with_multiple_strategies(input_grid, output_grid, debug=False):
    """
    Old fallback.

    This can still pick best-per-pair, but only as emergency fallback.
    The task-level family router should not rely on this for honest scoring.
    """
    candidates = []

    for strategy_name in LEGACY_PAIR_STRATEGIES:
        result = run_legacy_pair_strategy(strategy_name, input_grid, output_grid)

        if result is not None:
            candidates.append(result)

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item.get("exact", False),
            item.get("adjusted_score", 0),
            item.get("score", 0),
        ),
        reverse=True,
    )

    return candidates[0]


# ============================================================
# FIXED STRATEGY SCORING
# ============================================================

def score_fixed_apply_fn(
    family_name,
    task_rule,
    train_pairs,
    apply_fn,
):
    total_raw = 0
    total_adjusted = 0
    exact_count = 0
    results = []

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        output_grid = pair["output"]

        predicted = _safe_call(
            apply_fn,
            task_rule,
            input_grid,
            output_grid,
            pair_index,
        )

        raw_score = score_prediction(predicted, output_grid)
        penalty = shape_penalty(predicted, output_grid)
        adjusted = raw_score - penalty
        exact = predicted == output_grid

        if exact:
            exact_count += 1

        total_raw += raw_score
        total_adjusted += adjusted

        results.append(
            {
                "pair_index": pair_index,
                "predicted": predicted,
                "score": raw_score,
                "adjusted_score": adjusted,
                "exact": exact,
            }
        )

    return {
        "strategy": family_name,
        "family": family_name,
        "task_rule": task_rule,
        "rule": task_rule,
        "pair_count": len(train_pairs),
        "exact_count": exact_count,
        "total_raw_score": total_raw,
        "total_adjusted_score": total_adjusted,
        "results": results,
    }


def score_legacy_strategy_across_train(strategy_name, train_pairs):
    """
    Honest score for one fixed legacy strategy across all train pairs.
    """
    total_raw = 0
    total_adjusted = 0
    exact_count = 0
    results = []

    for pair_index, pair in enumerate(train_pairs):
        result = run_legacy_pair_strategy(
            strategy_name,
            pair["input"],
            pair["output"],
        )

        if result is None:
            predicted = None
            raw_score = 0
            adjusted = -10_000
            exact = False
        else:
            predicted = result.get("predicted")
            raw_score = result.get("score", 0)
            adjusted = result.get("adjusted_score", raw_score)
            exact = result.get("exact", False)

        if exact:
            exact_count += 1

        total_raw += raw_score
        total_adjusted += adjusted

        results.append(
            {
                "pair_index": pair_index,
                "predicted": predicted,
                "score": raw_score,
                "adjusted_score": adjusted,
                "exact": exact,
            }
        )

    return {
        "strategy": strategy_name,
        "pair_count": len(train_pairs),
        "exact_count": exact_count,
        "total_raw_score": total_raw,
        "total_adjusted_score": total_adjusted,
        "results": results,
    }


def choose_best_fixed_legacy_strategy(strategy_names, train_pairs):
    """
    Pick one fixed inner strategy for the whole task.
    """
    candidates = []

    for strategy_name in strategy_names:
        if LEGACY_PAIR_STRATEGIES.get(strategy_name) is None:
            continue

        score = score_legacy_strategy_across_train(strategy_name, train_pairs)
        candidates.append(score)

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item.get("exact_count", 0),
            item.get("total_adjusted_score", 0),
            item.get("total_raw_score", 0),
        ),
        reverse=True,
    )

    return candidates[0]


def leave_one_out_family(
    family_name,
    train_pairs,
    discover_fn,
    apply_fn,
):
    """
    Honest leave-one-out:
        - hide one pair
        - discover the rule from remaining pairs
        - apply that one discovered rule to hidden pair
    """
    if len(train_pairs) <= 1:
        return {
            "strategy": family_name,
            "family": family_name,
            "pair_count": 0,
            "exact_count": 0,
            "total_raw_score": 0,
            "total_adjusted_score": 0,
            "valid_prediction_count": 0,
            "missing_prediction_count": 0,
            "invalid_prediction_count": 0,
            "shape_exact_count": 0,
            "results": [],
        }

    total_raw = 0
    total_adjusted = 0
    exact_count = 0
    valid_prediction_count = 0
    missing_prediction_count = 0
    invalid_prediction_count = 0
    shape_exact_count = 0
    results = []

    for hidden_index in range(len(train_pairs)):
        visible_pairs = [
            pair
            for idx, pair in enumerate(train_pairs)
            if idx != hidden_index
        ]

        hidden_pair = train_pairs[hidden_index]

        task_rule = _safe_call(discover_fn, visible_pairs)

        predicted = _safe_call(
            apply_fn,
            task_rule,
            hidden_pair["input"],
            None,
            None,
        )

        valid_prediction = is_valid_arc_grid(predicted)
        prediction_missing = predicted is None
        invalid_prediction = predicted is not None and not valid_prediction

        if prediction_missing:
            missing_prediction_count += 1
        elif invalid_prediction:
            invalid_prediction_count += 1
        else:
            valid_prediction_count += 1

        evaluated_prediction = predicted if valid_prediction else None
        raw_score = score_prediction(evaluated_prediction, hidden_pair["output"])
        penalty = shape_penalty(evaluated_prediction, hidden_pair["output"])
        adjusted = raw_score - penalty
        exact = valid_prediction and predicted == hidden_pair["output"]
        shape_exact = (
            valid_prediction
            and grid_shape(predicted) == grid_shape(hidden_pair["output"])
        )

        if exact:
            exact_count += 1

        if shape_exact:
            shape_exact_count += 1

        total_raw += raw_score
        total_adjusted += adjusted

        results.append(
            {
                "pair_index": hidden_index,
                "predicted": predicted,
                "score": raw_score,
                "adjusted_score": adjusted,
                "exact": exact,
                "valid_prediction": valid_prediction,
                "prediction_missing": prediction_missing,
                "invalid_prediction": invalid_prediction,
                "shape_exact": shape_exact,
            }
        )

    return {
        "strategy": family_name,
        "family": family_name,
        "pair_count": len(train_pairs),
        "exact_count": exact_count,
        "total_raw_score": total_raw,
        "total_adjusted_score": total_adjusted,
        "valid_prediction_count": valid_prediction_count,
        "missing_prediction_count": missing_prediction_count,
        "invalid_prediction_count": invalid_prediction_count,
        "shape_exact_count": shape_exact_count,
        "results": results,
    }


def family_is_honest(scored_rule, loo_rule=None):
    """
    Gatekeeper.

    Perfect train is required.
    Leave-one-out is preferred, but small tasks may not have enough examples.
    """
    if scored_rule is None:
        return False

    if scored_rule.get("test_capable") is False:
        return False

    pair_count = scored_rule.get("pair_count", 0)
    exact_count = scored_rule.get("exact_count", 0)

    if pair_count == 0:
        return False

    if exact_count != pair_count:
        return False

    if loo_rule is None:
        return True

    loo_pair_count = loo_rule.get("pair_count", 0)
    loo_exact_count = loo_rule.get("exact_count", 0)
    loo_valid_count = loo_rule.get("valid_prediction_count", 0)

    if loo_pair_count == 0:
        return True

    return loo_valid_count == loo_pair_count and loo_exact_count > 0


# ============================================================
# FAMILY 1: VISUAL SYMBOLIC
# ============================================================

VISUAL_SYMBOLIC_FAMILY = "visual_symbolic_family"


def discover_visual_symbolic_family(train_pairs):
    if discover_visual_symbolic_rule_for_task is None:
        return None

    inner_rule = _safe_call(
        discover_visual_symbolic_rule_for_task,
        train_pairs,
    )

    if inner_rule is None:
        return None

    return {
        "family": VISUAL_SYMBOLIC_FAMILY,
        "rule_type": "visual_symbolic",
        "chosen_inner_strategy": "visual_symbolic_rule",
        "inner_rule": inner_rule,
    }


def apply_visual_symbolic_family(
    task_rule,
    input_grid,
    expected_grid=None,
    pair_index=None,
):
    if task_rule is None:
        return None

    if apply_visual_symbolic_rule is None:
        return None

    inner_rule = task_rule.get("inner_rule")

    result = _safe_call(
        apply_visual_symbolic_rule,
        inner_rule,
        input_grid,
    )

    predicted = extract_prediction(result)

    if predicted is not None:
        return predicted

    return result


def score_visual_symbolic_family(train_pairs):
    task_rule = discover_visual_symbolic_family(train_pairs)

    if task_rule is None:
        return None, None

    scored = score_fixed_apply_fn(
        VISUAL_SYMBOLIC_FAMILY,
        task_rule,
        train_pairs,
        apply_visual_symbolic_family,
    )

    if scored.get("exact_count") == scored.get("pair_count"):
        loo = leave_one_out_family(
            VISUAL_SYMBOLIC_FAMILY,
            train_pairs,
            discover_visual_symbolic_family,
            apply_visual_symbolic_family,
        )
    else:
        loo = None

    return scored, loo


# ============================================================
# FAMILY 2: PATTERN CANVAS
# ============================================================

PATTERN_CANVAS_FAMILY = "pattern_canvas_family"

PATTERN_CANVAS_STRATEGIES = [
    "pattern_rule",
    "pattern_expansion_rule",
]


def discover_pattern_canvas_family(train_pairs):
    best = choose_best_fixed_legacy_strategy(
        PATTERN_CANVAS_STRATEGIES,
        train_pairs,
    )

    if best is None:
        return None

    return {
        "family": PATTERN_CANVAS_FAMILY,
        "rule_type": "fixed_legacy_family_wrapper",
        "chosen_inner_strategy": best.get("strategy"),
        "inner_score": best,
        "inner_strategies_tested": list(PATTERN_CANVAS_STRATEGIES),
    }


def apply_pattern_canvas_family(
    task_rule,
    input_grid,
    expected_grid=None,
    pair_index=None,
):
    if task_rule is None:
        return None

    chosen_inner_strategy = task_rule.get("chosen_inner_strategy")

    if chosen_inner_strategy is None:
        return None

    result = run_legacy_pair_strategy(
        chosen_inner_strategy,
        input_grid,
        expected_grid,
    )

    if result is None:
        return None

    return result.get("predicted")


def score_pattern_canvas_family(train_pairs):
    task_rule = discover_pattern_canvas_family(train_pairs)

    if task_rule is None:
        return None, None

    scored = score_fixed_apply_fn(
        PATTERN_CANVAS_FAMILY,
        task_rule,
        train_pairs,
        apply_pattern_canvas_family,
    )

    if scored.get("exact_count") == scored.get("pair_count"):
        loo = leave_one_out_family(
            PATTERN_CANVAS_FAMILY,
            train_pairs,
            discover_pattern_canvas_family,
            apply_pattern_canvas_family,
        )
    else:
        loo = None

    return scored, loo


# ============================================================
# FAMILY 3: REGION OBJECT
# ============================================================

REGION_OBJECT_FAMILY = "region_object_family"

REGION_OBJECT_STRATEGIES = [
    "region_rule",
    "region_alignment_rule_v2",
    "object_grid_rule",
]


def score_learned_region_candidate(train_pairs, learned_rule):
    if learned_rule is None:
        return None

    if apply_learned_region_rule is None:
        return None

    total_raw = 0
    total_adjusted = 0
    exact_count = 0
    results = []

    for pair_index, pair in enumerate(train_pairs):
        result = _safe_call(
            apply_learned_region_rule,
            learned_rule,
            pair["input"],
        )

        predicted = extract_prediction(result)

        if predicted is None and result is not None:
            predicted = result

        raw_score = score_prediction(predicted, pair["output"])
        penalty = shape_penalty(predicted, pair["output"])
        adjusted = raw_score - penalty
        exact = predicted == pair["output"]

        if exact:
            exact_count += 1

        total_raw += raw_score
        total_adjusted += adjusted

        results.append(
            {
                "pair_index": pair_index,
                "predicted": predicted,
                "score": raw_score,
                "adjusted_score": adjusted,
                "exact": exact,
            }
        )

    return {
        "strategy": "learned_region_rule",
        "pair_count": len(train_pairs),
        "exact_count": exact_count,
        "total_raw_score": total_raw,
        "total_adjusted_score": total_adjusted,
        "results": results,
    }


def discover_region_object_family(train_pairs):
    candidates = []

    best_legacy = choose_best_fixed_legacy_strategy(
        REGION_OBJECT_STRATEGIES,
        train_pairs,
    )

    if best_legacy is not None:
        candidates.append(
            {
                "kind": "legacy",
                "strategy": best_legacy.get("strategy"),
                "score": best_legacy,
                "learned_rule": None,
            }
        )

    learned_region_rule = None

    if discover_learned_region_rule_for_task is not None:
        learned_region_rule = _safe_call(
            discover_learned_region_rule_for_task,
            train_pairs,
        )

    learned_score = score_learned_region_candidate(
        train_pairs,
        learned_region_rule,
    )

    if learned_score is not None:
        candidates.append(
            {
                "kind": "learned_region",
                "strategy": "learned_region_rule",
                "score": learned_score,
                "learned_rule": learned_region_rule,
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item["score"].get("exact_count", 0),
            item["score"].get("total_adjusted_score", 0),
            item["score"].get("total_raw_score", 0),
        ),
        reverse=True,
    )

    best = candidates[0]

    return {
        "family": REGION_OBJECT_FAMILY,
        "rule_type": "fixed_region_object_wrapper",
        "chosen_inner_strategy": best["strategy"],
        "chosen_inner_kind": best["kind"],
        "learned_region_rule": best.get("learned_rule"),
        "inner_score": best["score"],
        "inner_strategies_tested": list(REGION_OBJECT_STRATEGIES) + ["learned_region_rule"],
    }


def apply_region_object_family(
    task_rule,
    input_grid,
    expected_grid=None,
    pair_index=None,
):
    if task_rule is None:
        return None

    chosen_inner_strategy = task_rule.get("chosen_inner_strategy")

    if chosen_inner_strategy == "learned_region_rule":
        learned_rule = task_rule.get("learned_region_rule")

        if learned_rule is None:
            return None

        if apply_learned_region_rule is None:
            return None

        result = _safe_call(
            apply_learned_region_rule,
            learned_rule,
            input_grid,
        )

        predicted = extract_prediction(result)

        if predicted is not None:
            return predicted

        return result

    result = run_legacy_pair_strategy(
        chosen_inner_strategy,
        input_grid,
        expected_grid,
    )

    if result is None:
        return None

    return result.get("predicted")


def score_region_object_family(train_pairs):
    task_rule = discover_region_object_family(train_pairs)

    if task_rule is None:
        return None, None

    scored = score_fixed_apply_fn(
        REGION_OBJECT_FAMILY,
        task_rule,
        train_pairs,
        apply_region_object_family,
    )

    if scored.get("exact_count") == scored.get("pair_count"):
        loo = leave_one_out_family(
            REGION_OBJECT_FAMILY,
            train_pairs,
            discover_region_object_family,
            apply_region_object_family,
        )
    else:
        loo = None

    return scored, loo


# ============================================================
# FAMILY 4: COMPOSITION LAYOUT
# ============================================================

COMPOSITION_LAYOUT_FAMILY = "composition_layout_family"

COMPOSITION_LAYOUT_STRATEGIES = [
    "motif_layout_rule",
    "seed_placement_expansion_rule",
]


def _load_motif_path_layout_fns():
    learn_fn = _optional_import(
        lambda: __import__(
            "reasoning.motif_path_layout_rule",
            fromlist=["learn_motif_path_layout_rule"],
        ).learn_motif_path_layout_rule
    )

    apply_fn = _optional_import(
        lambda: __import__(
            "reasoning.motif_path_layout_rule",
            fromlist=["apply_motif_path_layout_rule"],
        ).apply_motif_path_layout_rule
    )

    guesses_fn = _optional_import(
        lambda: __import__(
            "reasoning.motif_path_layout_rule",
            fromlist=["generate_motif_path_layout_guesses"],
        ).generate_motif_path_layout_guesses
    )

    return learn_fn, apply_fn, guesses_fn


def score_multi_seed_candidate(train_pairs, multi_seed_rule):
    if multi_seed_rule is None:
        return None

    total_raw = 0
    total_adjusted = 0
    exact_count = 0
    results = []

    for pair_index, pair in enumerate(train_pairs):
        predicted = None

        if apply_multi_seed_composition_rule_for_train_pair is not None:
            result = _safe_call(
                apply_multi_seed_composition_rule_for_train_pair,
                multi_seed_rule,
                pair["input"],
                pair_index,
            )

            predicted = extract_prediction(result)

            if predicted is None and result is not None:
                predicted = result

        if predicted is None and apply_multi_seed_composition_rule is not None:
            result = _safe_call(
                apply_multi_seed_composition_rule,
                multi_seed_rule,
                pair["input"],
            )

            predicted = extract_prediction(result)

            if predicted is None and result is not None:
                predicted = result

        raw_score = score_prediction(predicted, pair["output"])
        penalty = shape_penalty(predicted, pair["output"])
        adjusted = raw_score - penalty
        exact = predicted == pair["output"]

        if exact:
            exact_count += 1

        total_raw += raw_score
        total_adjusted += adjusted

        results.append(
            {
                "pair_index": pair_index,
                "predicted": predicted,
                "score": raw_score,
                "adjusted_score": adjusted,
                "exact": exact,
            }
        )

    return {
        "strategy": "multi_seed_composition_rule",
        "pair_count": len(train_pairs),
        "exact_count": exact_count,
        "total_raw_score": total_raw,
        "total_adjusted_score": total_adjusted,
        "results": results,
    }


def score_motif_path_layout_candidate(train_pairs, motif_path_rule):
    if motif_path_rule is None:
        return None

    learn_fn, apply_fn, guesses_fn = _load_motif_path_layout_fns()

    if apply_fn is None:
        return None

    total_raw = 0
    total_adjusted = 0
    exact_count = 0
    results = []

    for pair_index, pair in enumerate(train_pairs):
        predicted = _safe_call(
            apply_fn,
            motif_path_rule,
            pair["input"],
            0,
        )

        raw_score = score_prediction(predicted, pair["output"])
        penalty = shape_penalty(predicted, pair["output"])
        adjusted = raw_score - penalty
        exact = predicted == pair["output"]

        if exact:
            exact_count += 1

        total_raw += raw_score
        total_adjusted += adjusted

        results.append(
            {
                "pair_index": pair_index,
                "predicted": predicted,
                "score": raw_score,
                "adjusted_score": adjusted,
                "exact": exact,
            }
        )

    return {
        "strategy": "motif_path_layout_rule",
        "pair_count": len(train_pairs),
        "exact_count": exact_count,
        "total_raw_score": total_raw,
        "total_adjusted_score": total_adjusted,
        "results": results,
    }


def discover_composition_layout_family(train_pairs):
    candidates = []

    # ------------------------------------------------------------
    # New learned motif path layout rule.
    # Prefer this over old motif_layout_rule when it solves train.
    # ------------------------------------------------------------
    learn_motif_path_layout_rule, apply_motif_path_layout_rule, _ = (
        _load_motif_path_layout_fns()
    )

    motif_path_rule = None

    if learn_motif_path_layout_rule is not None:
        motif_path_rule = _safe_call(
            learn_motif_path_layout_rule,
            train_pairs,
        )

    motif_path_score = score_motif_path_layout_candidate(
        train_pairs,
        motif_path_rule,
    )

    if motif_path_score is not None:
        candidates.append(
            {
                "kind": "learned_motif_path",
                "strategy": "motif_path_layout_rule",
                "score": motif_path_score,
                "motif_path_rule": motif_path_rule,
                "multi_seed_rule": None,
            }
        )

    # ------------------------------------------------------------
    # Existing multi-seed learned rule.
    # ------------------------------------------------------------
    multi_seed_rule = None

    if discover_multi_seed_composition_rule_for_task is not None:
        multi_seed_rule = _safe_call(
            discover_multi_seed_composition_rule_for_task,
            train_pairs,
        )

    multi_seed_score = score_multi_seed_candidate(
        train_pairs,
        multi_seed_rule,
    )

    if multi_seed_score is not None:
        candidates.append(
            {
                "kind": "multi_seed",
                "strategy": "multi_seed_composition_rule",
                "score": multi_seed_score,
                "motif_path_rule": None,
                "multi_seed_rule": multi_seed_rule,
            }
        )

    # ------------------------------------------------------------
    # Old fixed legacy strategies.
    # Kept as fallback only.
    # ------------------------------------------------------------
    best_legacy = choose_best_fixed_legacy_strategy(
        COMPOSITION_LAYOUT_STRATEGIES,
        train_pairs,
    )

    if best_legacy is not None:
        candidates.append(
            {
                "kind": "legacy",
                "strategy": best_legacy.get("strategy"),
                "score": best_legacy,
                "motif_path_rule": None,
                "multi_seed_rule": None,
            }
        )

    if not candidates:
        return None

    def candidate_rank(item):
        kind_bonus = {
            "learned_motif_path": 3,
            "multi_seed": 2,
            "legacy": 0,
        }.get(item["kind"], 0)

        score = item["score"]

        return (
            score.get("exact_count", 0),
            kind_bonus,
            score.get("total_adjusted_score", 0),
            score.get("total_raw_score", 0),
        )

    candidates.sort(key=candidate_rank, reverse=True)
    best = candidates[0]

    if best["kind"] == "learned_motif_path":
        rule_type = "learned_motif_path_layout_wrapper"
    elif best["kind"] == "multi_seed":
        rule_type = "learned_multi_seed_composition_wrapper"
    else:
        rule_type = "fixed_composition_layout_wrapper"

    return {
        "family": COMPOSITION_LAYOUT_FAMILY,
        "rule_type": rule_type,
        "chosen_inner_strategy": best["strategy"],
        "chosen_inner_kind": best["kind"],
        "motif_path_rule": best.get("motif_path_rule"),
        "multi_seed_rule": best.get("multi_seed_rule"),
        "inner_score": best["score"],
        "inner_strategies_tested": [
            "motif_path_layout_rule",
            "multi_seed_composition_rule",
        ] + list(COMPOSITION_LAYOUT_STRATEGIES),
    }


def apply_composition_layout_family(
    task_rule,
    input_grid,
    expected_grid=None,
    pair_index=None,
):
    if task_rule is None:
        return None

    chosen_inner_strategy = task_rule.get("chosen_inner_strategy")

    # ------------------------------------------------------------
    # New learned motif path layout.
    # ------------------------------------------------------------
    if chosen_inner_strategy == "motif_path_layout_rule":
        motif_path_rule = task_rule.get("motif_path_rule")

        if motif_path_rule is None:
            return None

        _, apply_fn, _ = _load_motif_path_layout_fns()

        if apply_fn is None:
            return None

        result = _safe_call(
            apply_fn,
            motif_path_rule,
            input_grid,
            0,
        )

        predicted = extract_prediction(result)

        if predicted is not None:
            return predicted

        return result

    # ------------------------------------------------------------
    # Existing multi-seed learned rule.
    # ------------------------------------------------------------
    if chosen_inner_strategy == "multi_seed_composition_rule":
        multi_seed_rule = task_rule.get("multi_seed_rule")

        if multi_seed_rule is None:
            return None

        if expected_grid is not None and apply_multi_seed_composition_rule_for_train_pair is not None:
            result = _safe_call(
                apply_multi_seed_composition_rule_for_train_pair,
                multi_seed_rule,
                input_grid,
                pair_index,
            )

            predicted = extract_prediction(result)

            if predicted is not None:
                return predicted

            if result is not None:
                return result

        if apply_multi_seed_composition_rule is not None:
            result = _safe_call(
                apply_multi_seed_composition_rule,
                multi_seed_rule,
                input_grid,
            )

            predicted = extract_prediction(result)

            if predicted is not None:
                return predicted

            if result is not None:
                return result

        return None

    # ------------------------------------------------------------
    # Old legacy fallback.
    # ------------------------------------------------------------
    result = run_legacy_pair_strategy(
        chosen_inner_strategy,
        input_grid,
        expected_grid,
    )

    if result is None:
        return None

    return result.get("predicted")


def score_composition_layout_family(train_pairs):
    task_rule = discover_composition_layout_family(train_pairs)

    if task_rule is None:
        return None, None

    scored = score_fixed_apply_fn(
        COMPOSITION_LAYOUT_FAMILY,
        task_rule,
        train_pairs,
        apply_composition_layout_family,
    )

    if scored.get("exact_count") == scored.get("pair_count"):
        loo = leave_one_out_family(
            COMPOSITION_LAYOUT_FAMILY,
            train_pairs,
            discover_composition_layout_family,
            apply_composition_layout_family,
        )
    else:
        loo = None

    return scored, loo


# ============================================================
# FAMILY 5: ANCHOR REPAIR
# ============================================================

ANCHOR_REPAIR_FAMILY = "anchor_repair_family"


def task_looks_like_anchor_repair(train_pairs):
    """
    Fast pre-check.

    Anchor repair tasks are small-color tasks:
        background + structure + anchor

    Do not run this on big noisy mask/pattern tasks like 0934a4d8.
    """
    if not train_pairs:
        return False

    for pair in train_pairs:
        input_grid = pair.get("input")
        output_grid = pair.get("output")

        if input_grid is None or output_grid is None:
            return False

        input_colors = set()
        output_colors = set()

        for row in input_grid:
            input_colors.update(row)

        for row in output_grid:
            output_colors.update(row)

        # Anchor repair should not be running on huge many-color quilt tasks.
        if len(input_colors) > 4:
            return False

        # Output should usually drop the anchor color, not introduce many colors.
        if len(output_colors) > 3:
            return False

    return True


def discover_anchor_repair_family(train_pairs):
    if discover_anchor_repair_rule_for_task is None:
        return None

    if not task_looks_like_anchor_repair(train_pairs):
        return None

    inner_rule = _safe_call(
        discover_anchor_repair_rule_for_task,
        train_pairs,
    )

    if inner_rule is None:
        return None

    return {
        "family": ANCHOR_REPAIR_FAMILY,
        "rule_type": "anchor_repair",
        "chosen_inner_strategy": "anchor_repair_rule",
        "inner_rule": inner_rule,
    }


def apply_anchor_repair_family(
    task_rule,
    input_grid,
    expected_grid=None,
    pair_index=None,
):
    if task_rule is None:
        return None

    if apply_anchor_repair_rule is None:
        return None

    inner_rule = task_rule.get("inner_rule")

    if inner_rule is None:
        return None

    result = _safe_call(
        apply_anchor_repair_rule,
        inner_rule,
        input_grid,
    )

    predicted = extract_prediction(result)

    if predicted is not None:
        return predicted

    return result


def score_anchor_repair_family(train_pairs):
    task_rule = discover_anchor_repair_family(train_pairs)

    if task_rule is None:
        return None, None

    scored = score_fixed_apply_fn(
        ANCHOR_REPAIR_FAMILY,
        task_rule,
        train_pairs,
        apply_anchor_repair_family,
    )

    if scored.get("exact_count") == scored.get("pair_count"):
        loo = leave_one_out_family(
            ANCHOR_REPAIR_FAMILY,
            train_pairs,
            discover_anchor_repair_family,
            apply_anchor_repair_family,
        )
    else:
        loo = None

    return scored, loo


# ============================================================
# FAMILY TABLE
# ============================================================

FAMILY_SCORERS = [
    score_visual_symbolic_family,
    score_pattern_canvas_family,
    score_region_object_family,
    score_composition_layout_family,
    score_anchor_repair_family,
]

FAMILY_APPLIERS = {
    VISUAL_SYMBOLIC_FAMILY: apply_visual_symbolic_family,
    PATTERN_CANVAS_FAMILY: apply_pattern_canvas_family,
    REGION_OBJECT_FAMILY: apply_region_object_family,
    COMPOSITION_LAYOUT_FAMILY: apply_composition_layout_family,
    ANCHOR_REPAIR_FAMILY: apply_anchor_repair_family,
}


# ============================================================
# MAIN TASK-LEVEL ROUTER
# ============================================================

def choose_task_level_strategy(train_pairs, debug=False):
    """
    Choose one family for the whole task.
    """
    if not train_pairs:
        return {
            "best_strategy": None,
            "task_rule": None,
            "rule": None,
            "strategy_stats": {},
            "family_scores": [],
            "features": {},
        }

    features = detect_task_features(train_pairs)

    family_scores = []
    strategy_stats = {}

    for score_fn in FAMILY_SCORERS:
        result = _safe_call(score_fn, train_pairs)

        if result is None:
            continue

        scored, loo = result

        if scored is None:
            continue

        family_name = scored.get("family") or scored.get("strategy")
        task_rule = scored.get("task_rule") or scored.get("rule") or {}

        family_applier = FAMILY_APPLIERS.get(family_name)
        test_probe = None

        if family_applier is not None and train_pairs:
            test_probe = _safe_call(
                family_applier,
                task_rule,
                train_pairs[0].get("input"),
                None,
                None,
            )

        test_capable = is_valid_arc_grid(test_probe)
        scored["test_capable"] = test_capable

        honest = family_is_honest(scored, loo)

        scored["honest"] = honest
        scored["leave_one_out"] = loo

        pair_count = scored.get("pair_count", 0)
        exact_count = scored.get("exact_count", 0)

        loo_pair_count = loo.get("pair_count", 0) if loo else 0
        loo_exact_count = loo.get("exact_count", 0) if loo else 0
        loo_valid_count = loo.get("valid_prediction_count", 0) if loo else 0
        loo_missing_count = loo.get("missing_prediction_count", 0) if loo else 0
        loo_invalid_count = loo.get("invalid_prediction_count", 0) if loo else 0
        loo_shape_exact_count = loo.get("shape_exact_count", 0) if loo else 0

        strategy_stats[family_name] = {
            "pair_count": pair_count,
            "exact_count": exact_count,
            "total_raw_score": scored.get("total_raw_score", 0),
            "total_adjusted_score": scored.get("total_adjusted_score", 0),
            "loo_pair_count": loo_pair_count,
            "loo_exact_count": loo_exact_count,
            "loo_valid_prediction_count": loo_valid_count,
            "loo_missing_prediction_count": loo_missing_count,
            "loo_invalid_prediction_count": loo_invalid_count,
            "loo_shape_exact_count": loo_shape_exact_count,
            "test_capable": test_capable,
            "honest": honest,
            "chosen_inner_strategy": task_rule.get("chosen_inner_strategy"),
            "rule_type": task_rule.get("rule_type"),
        }

        family_scores.append(scored)

    if not family_scores:
        return {
            "best_strategy": None,
            "task_rule": None,
            "rule": None,
            "strategy_stats": strategy_stats,
            "family_scores": [],
            "features": features,
        }

    def ranking_key(item):
        loo = item.get("leave_one_out") or {}
        task_rule = item.get("task_rule") or item.get("rule") or {}

        rule_type = task_rule.get("rule_type")
        task_level_bonus = 1 if rule_type in {
    "visual_symbolic",
    "fixed_region_object_wrapper",
    "fixed_composition_layout_wrapper",
    "learned_motif_path_layout_wrapper",
    "learned_multi_seed_composition_wrapper",
} else 0

        return (
            item.get("test_capable", False),
            item.get("honest", False),
            loo.get("exact_count", 0),
            loo.get("shape_exact_count", 0),
            loo.get("valid_prediction_count", 0),
            item.get("exact_count", 0),
            task_level_bonus,
            item.get("total_adjusted_score", 0),
            item.get("total_raw_score", 0),
        )

    family_scores.sort(key=ranking_key, reverse=True)
    best = family_scores[0]

    best_family = best.get("family") or best.get("strategy")
    task_rule = best.get("task_rule") or best.get("rule")

    if debug:
        print("\nTASK ROUTER FAMILY SCORES")
        print("-" * 60)

        for item in family_scores:
            loo = item.get("leave_one_out") or {}
            task_rule_for_print = item.get("task_rule") or item.get("rule") or {}

            print(
                f"{item.get('family')}: "
                f"inner={task_rule_for_print.get('chosen_inner_strategy')} "
                f"exact={item.get('exact_count')}/{item.get('pair_count')} "
                f"loo={loo.get('exact_count', 0)}/{loo.get('pair_count', 0)} "
                f"loo_valid={loo.get('valid_prediction_count', 0)} "
                f"test_capable={item.get('test_capable')} "
                f"honest={item.get('honest')} "
                f"adj={item.get('total_adjusted_score')}"
            )

    return {
        "best_strategy": best_family,
        "task_rule": task_rule,
        "rule": task_rule,
        "strategy_stats": strategy_stats,
        "family_scores": family_scores,
        "features": features,
    }


# ============================================================
# APPLY CHOSEN TASK RULE
# ============================================================

def apply_task_rule_to_input(
    strategy_name,
    task_rule,
    input_grid,
    expected_grid=None,
    pair_index=None,
):
    if strategy_name is None:
        return None

    if strategy_name in FAMILY_APPLIERS:
        return _safe_call(
            FAMILY_APPLIERS[strategy_name],
            task_rule,
            input_grid,
            expected_grid,
            pair_index,
        )

    result = run_legacy_pair_strategy(
        strategy_name,
        input_grid,
        expected_grid,
    )

    if result is None:
        return None

    return result.get("predicted")


def score_task_rule_prediction(
    strategy_name,
    task_rule,
    input_grid,
    expected_grid,
    pair_index=None,
):
    predicted = apply_task_rule_to_input(
        strategy_name=strategy_name,
        task_rule=task_rule,
        input_grid=input_grid,
        expected_grid=expected_grid,
        pair_index=pair_index,
    )

    return normalize_prediction_result(
        strategy=strategy_name,
        predicted=predicted,
        expected=expected_grid,
        raw_result={
            "pair_index": pair_index,
            "task_rule": task_rule,
        },
    )


# ============================================================
# TEST-ONLY SPECIAL HELPER
# ============================================================

def try_anchor_compass_merge_for_test_pair(
    task,
    test_pair,
    test_index=None,
    debug=False,
):
    """
    Kept for compatibility with run_solver.py.

    This should later move into composition_layout_family
    or visual_symbolic_family.
    """
    if predict_anchor_compass_merge_for_pair is None:
        return None

    result = _safe_call(
        predict_anchor_compass_merge_for_pair,
        task,
        test_pair,
        test_index,
        debug,
    )

    if result is None:
        return None

    predicted = extract_prediction(result)

    if predicted is None:
        return None

    if isinstance(result, dict):
        result = dict(result)
    else:
        result = {}

    result["strategy"] = "anchor_compass_merge_rule"
    result["predicted"] = predicted
    result["prediction"] = predicted

    return result
