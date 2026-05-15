# strategies/ring_blob_rule_synthesizer_strategy.py
"""
Ring Blob Rule Synthesizer Strategy

This is the router-facing wrapper around:

    reasoning/ring_blob_rule_synthesizer.py

Purpose:
    Make the all-train ring/blob synthesizer available as a normal
    strategy family.

Important:
    This does NOT pick one closest train pair.
    It learns reusable pieces from all train pairs and composes the
    test output.

Expected strategy name:
    ring_blob_rule_synthesizer
"""

from reasoning.archive.ring_blob_rule_synthesizer import (
    learn_ring_blob_rule_synthesizer,
    predict_with_ring_blob_rule_synthesizer,
)


# ============================================================
# Basic safety helpers
# ============================================================

def is_valid_grid(grid):
    if not isinstance(grid, list):
        return False

    if not grid:
        return False

    if not all(isinstance(row, list) for row in grid):
        return False

    width = len(grid[0])

    if width == 0:
        return False

    for row in grid:
        if len(row) != width:
            return False

    return True


def is_valid_train_pair(pair):
    if not isinstance(pair, dict):
        return False

    if "input" not in pair or "output" not in pair:
        return False

    if not is_valid_grid(pair["input"]):
        return False

    if not is_valid_grid(pair["output"]):
        return False

    return True


def normalize_train_pairs(train_pairs):
    """
    Keep only valid train pairs.

    This prevents one bad task record from crashing the whole router.
    """
    clean_pairs = []

    for pair in train_pairs:
        if is_valid_train_pair(pair):
            clean_pairs.append(pair)

    return clean_pairs


# ============================================================
# Strategy gating
# ============================================================

def task_looks_like_ring_blob_problem(train_pairs):
    """
    Conservative gate.

    We only want this strategy to run on tasks that are likely to have:
        - ring-like objects
        - blob-like objects
        - symbolic output mapping

    The deeper ring/blob learner still decides the details.
    This function only prevents the strategy from firing everywhere.
    """
    clean_pairs = normalize_train_pairs(train_pairs)

    if not clean_pairs:
        return False

    try:
        rule = learn_ring_blob_rule_synthesizer(clean_pairs)
    except Exception:
        return False

    records = rule.get("records", [])

    if not records:
        return False

    ring_scene_count = 0
    useful_scene_count = 0

    for record in records:
        signature = record.get("signature", {})

        ring_count = signature.get("ring_count", 0)
        blob_count = signature.get("blob_count", 0)

        if ring_count > 0:
            ring_scene_count += 1

        if ring_count > 0 and blob_count > 0:
            useful_scene_count += 1

    # Require at least most training pairs to have rings/blobs.
    return useful_scene_count >= max(1, len(records) - 1)


# ============================================================
# Public strategy entry point
# ============================================================

def solve_with_ring_blob_rule_synthesizer(train_pairs, test_input):
    """
    Main function used by the task router.

    Args:
        train_pairs:
            [
                {"input": grid, "output": grid},
                ...
            ]

        test_input:
            grid

    Returns:
        strategy result dict, or None if not applicable.
    """
    clean_pairs = normalize_train_pairs(train_pairs)

    if not clean_pairs:
        return None

    if not is_valid_grid(test_input):
        return None

    try:
        rule = learn_ring_blob_rule_synthesizer(clean_pairs)
    except Exception as exc:
        return {
            "strategy": "ring_blob_rule_synthesizer",
            "prediction": None,
            "exact": False,
            "score": -10**9,
            "mode": "learn_failed",
            "error": repr(exc),
        }

    records = rule.get("records", [])

    if not records:
        return None

    # Gate after learning so we use the real extracted records.
    ring_scene_count = 0
    useful_scene_count = 0

    for record in records:
        signature = record.get("signature", {})

        ring_count = signature.get("ring_count", 0)
        blob_count = signature.get("blob_count", 0)

        if ring_count > 0:
            ring_scene_count += 1

        if ring_count > 0 and blob_count > 0:
            useful_scene_count += 1

    if useful_scene_count < max(1, len(records) - 1):
        return None

    try:
        result = predict_with_ring_blob_rule_synthesizer(
            rule,
            test_input,
        )
    except Exception as exc:
        return {
            "strategy": "ring_blob_rule_synthesizer",
            "prediction": None,
            "exact": False,
            "score": -10**9,
            "mode": "predict_failed",
            "error": repr(exc),
        }

    prediction = result.get("prediction")

    if prediction is None:
        return None

    return {
        "strategy": "ring_blob_rule_synthesizer",
        "prediction": prediction,
        "exact": False,
        "score": 0,
        "mode": "all_train_composed_ring_blob_rule",
        "details": {
            "base_train_pair": result.get("base_train_pair"),
            "base_reason": result.get("base_reason"),
            "test_signature": result.get("test_signature"),
            "base_signature": result.get("base_signature"),
            "notes": result.get("notes", []),
        },
    }


# ============================================================
# Compatibility alias
# ============================================================

def apply_ring_blob_rule_synthesizer(train_pairs, test_input):
    """
    Alias in case the router prefers apply_* naming.
    """
    return solve_with_ring_blob_rule_synthesizer(train_pairs, test_input)