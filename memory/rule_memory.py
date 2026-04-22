# memory/rule_memory.py

import json
import os
from copy import deepcopy


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
MEMORY_PATH = os.path.join(MEMORY_DIR, "rule_memory.json")


def ensure_memory_file():
    os.makedirs(MEMORY_DIR, exist_ok=True)

    if not os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "history": [],
                    "patterns": [],
                    "rules": []
                },
                f,
                indent=2
            )


def load_rule_memory():
    ensure_memory_file()

    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "history" not in data:
        data["history"] = []
    if "patterns" not in data:
        data["patterns"] = []
    if "rules" not in data:
        data["rules"] = []

    return data


def save_rule_memory(data):
    ensure_memory_file()

    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def detect_basic_features(train_pairs):
    """
    Very lightweight pattern summary.
    This is not the solver.
    This is just metadata/hints for future routing.
    """
    features = {
        "pair_count": len(train_pairs),
        "has_divider_color_4": False,
        "has_anchor_color_5": False,
        "output_widths": [],
        "output_heights": [],
    }

    for pair in train_pairs:
        inp = pair.get("input", [])
        out = pair.get("output", [])

        # output shape
        if out:
            features["output_heights"].append(len(out))
            features["output_widths"].append(len(out[0]) if out[0] else 0)

        # detect divider 4 and anchor 5
        has_4 = any(4 in row for row in inp)
        has_5 = any(5 in row for row in inp)

        if has_4:
            features["has_divider_color_4"] = True
        if has_5:
            features["has_anchor_color_5"] = True

    return features


def upsert_pattern(memory, pattern_entry):
    """
    Add a pattern entry only if an equivalent one is not already present.
    """
    for existing in memory["patterns"]:
        if (
            existing.get("suggested_strategy") == pattern_entry.get("suggested_strategy")
            and existing.get("features") == pattern_entry.get("features")
        ):
            return

    memory["patterns"].append(pattern_entry)


def append_history(memory, history_entry):
    """
    Always append history; history is a log.
    """
    memory["history"].append(history_entry)


def save_successful_task_memory(task_id, strategy_name, train_pairs, exact_count, total_pairs):
    """
    Save a success record for a task run.
    Intended to be called when a strategy is chosen and has good train performance.
    """
    memory = load_rule_memory()
    features = detect_basic_features(train_pairs)

    history_entry = {
        "task_id": task_id,
        "strategy": strategy_name,
        "exact_count": exact_count,
        "total_pairs": total_pairs,
        "result": "exact" if exact_count == total_pairs else "partial",
        "features": deepcopy(features),
    }

    append_history(memory, history_entry)

    # Only store a reusable pattern if the task solved all train pairs exactly
    if exact_count == total_pairs:
        pattern_entry = {
            "name": f"{strategy_name}_{task_id}",
            "features": deepcopy(features),
            "suggested_strategy": strategy_name,
            "source_task": task_id,
        }
        upsert_pattern(memory, pattern_entry)

    save_rule_memory(memory)


def get_memory_suggested_strategies(train_pairs):
    """
    Return strategies suggested by stored pattern memory.
    This should GUIDE the router, not force it.
    """
    memory = load_rule_memory()
    current_features = detect_basic_features(train_pairs)

    suggestions = []

    for pattern in memory.get("patterns", []):
        if pattern.get("features") == current_features:
            suggestions.append(pattern.get("suggested_strategy"))

    return suggestions