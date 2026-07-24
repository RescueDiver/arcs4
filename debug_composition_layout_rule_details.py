# debug_composition_layout_rule_details.py
import json
import os
import pprint
import sys
import traceback

from core.grid_utils import print_grid
from OLD.arc_visualizer import show_three_grids

from reasoning.task_router import (
    choose_task_level_strategy,
    apply_task_rule_to_input,
    score_prediction,
)


# ============================================================
# LOAD TASK
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def unwrap_task(raw, task_id):
    if "train" in raw:
        return raw

    if task_id in raw:
        return raw[task_id]

    if isinstance(raw, dict) and len(raw) == 1:
        return raw[next(iter(raw))]

    raise KeyError(f"Could not find task {task_id}")


def load_task(task_id_or_path):
    base_dir = os.path.dirname(__file__)
    value = task_id_or_path.strip().strip('"')

    if value.endswith(".json") and os.path.exists(value):
        raw = load_json(value)
        task_id = os.path.splitext(os.path.basename(value))[0]
        return task_id, unwrap_task(raw, task_id)

    failure_path = os.path.join(
        base_dir,
        "data_failures",
        "extracted_tasks",
        value + ".json",
    )

    if os.path.exists(failure_path):
        raw = load_json(failure_path)
        return value, unwrap_task(raw, value)

    data_path = os.path.join(base_dir, "data", "data.json")

    if os.path.exists(data_path):
        data = load_json(data_path)
        if value in data:
            return value, data[value]

    raise FileNotFoundError(value)


# ============================================================
# SAFE PRINT HELPERS
# ============================================================

def grid_shape(grid):
    if grid is None:
        return "None"

    h = len(grid)
    w = len(grid[0]) if h else 0
    return f"{h}x{w}"


def safe_repr(value, max_len=5000):
    text = pprint.pformat(value, width=120, compact=False)

    if len(text) > max_len:
        return text[:max_len] + "\n... <TRUNCATED> ..."

    return text


def print_title(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_section(title):
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


def is_grid(value):
    if not isinstance(value, list):
        return False

    if not value:
        return False

    if not all(isinstance(row, list) for row in value):
        return False

    return True


def summarize_value(value, indent=0, max_depth=4, name="root"):
    """
    Print nested rule objects without dumping giant grids unless needed.
    """
    prefix = " " * indent

    if max_depth <= 0:
        print(f"{prefix}{name}: <max depth reached>")
        return

    if is_grid(value):
        print(f"{prefix}{name}: GRID {grid_shape(value)}")
        return

    if isinstance(value, dict):
        print(f"{prefix}{name}: dict keys={list(value.keys())}")

        for key, child in value.items():
            if key in ("predicted", "prediction", "input", "output", "grid"):
                if is_grid(child):
                    print(f"{prefix}  {key}: GRID {grid_shape(child)}")
                else:
                    print(f"{prefix}  {key}: {type(child).__name__}")
                continue

            summarize_value(
                child,
                indent=indent + 2,
                max_depth=max_depth - 1,
                name=str(key),
            )

        return

    if isinstance(value, list):
        print(f"{prefix}{name}: list len={len(value)}")

        for idx, item in enumerate(value[:10]):
            summarize_value(
                item,
                indent=indent + 2,
                max_depth=max_depth - 1,
                name=f"[{idx}]",
            )

        if len(value) > 10:
            print(f"{prefix}  ... {len(value) - 10} more items")

        return

    print(f"{prefix}{name}: {repr(value)}")


# ============================================================
# ROUTER RESULT NORMALIZATION
# ============================================================

def normalize_router_result(router_result):
    """
    Different router versions may return either:
      - rule dict directly
      - result dict with task_rule
      - tuple/list where one item is the rule/result

    This function tries to extract:
      chosen_strategy
      task_rule
      strategy_stats
      raw
    """
    chosen_strategy = None
    task_rule = None
    strategy_stats = None

    raw = router_result

    if isinstance(router_result, dict):
        chosen_strategy = (
            router_result.get("chosen_strategy")
            or router_result.get("strategy")
            or router_result.get("family")
        )

        task_rule = (
            router_result.get("task_rule")
            or router_result.get("rule")
            or router_result.get("learned_rule")
            or router_result
        )

        strategy_stats = (
            router_result.get("strategy_stats")
            or router_result.get("stats")
            or router_result.get("all_results")
            or router_result.get("candidates")
        )

    elif isinstance(router_result, (tuple, list)):
        for item in router_result:
            if isinstance(item, str) and chosen_strategy is None:
                chosen_strategy = item

            if isinstance(item, dict):
                if task_rule is None:
                    if (
                        "family" in item
                        or "rule_type" in item
                        or "task_rule" in item
                        or "rule" in item
                    ):
                        task_rule = (
                            item.get("task_rule")
                            or item.get("rule")
                            or item
                        )

                if strategy_stats is None:
                    maybe_stats = (
                        item.get("strategy_stats")
                        or item.get("stats")
                        or item.get("all_results")
                        or item.get("candidates")
                    )
                    if maybe_stats is not None:
                        strategy_stats = maybe_stats

        if task_rule is None:
            for item in router_result:
                if isinstance(item, dict):
                    task_rule = item
                    break

    return {
        "chosen_strategy": chosen_strategy,
        "task_rule": task_rule,
        "strategy_stats": strategy_stats,
        "raw": raw,
    }


# ============================================================
# RULE INSPECTION
# ============================================================

def print_rule_summary(task_rule):
    print_title("TASK RULE SUMMARY")

    if task_rule is None:
        print("No task rule found.")
        return

    if not isinstance(task_rule, dict):
        print(f"Task rule type: {type(task_rule)}")
        print(safe_repr(task_rule))
        return

    print(f"family   : {task_rule.get('family')}")
    print(f"rule_type: {task_rule.get('rule_type')}")
    print(f"keys     : {list(task_rule.keys())}")

    print_section("TOP LEVEL RULE OBJECT")
    summarize_value(task_rule, max_depth=5)

    interesting_keys = [
        "multi_seed_rule",
        "learned_rule",
        "composition_rule",
        "layout_rule",
        "placement_rule",
        "chosen_inner_kind",
        "chosen_inner_strategy",
        "inner_score",
        "inner_strategies_tested",
        "examples",
        "placements",
        "events",
        "objects",
        "motifs",
    ]

    for key in interesting_keys:
        if key in task_rule:
            print_section(f"DETAIL: {key}")
            summarize_value(task_rule[key], max_depth=8, name=key)
            print()
            print(safe_repr(task_rule[key], max_len=12000))


def print_strategy_stats(strategy_stats):
    print_title("STRATEGY STATS / CANDIDATES")

    if strategy_stats is None:
        print("No strategy stats found.")
        return

    summarize_value(strategy_stats, max_depth=6)
    print()
    print(safe_repr(strategy_stats, max_len=12000))


# ============================================================
# APPLY CHECKS
# ============================================================

def apply_rule(task_rule, input_grid, expected_grid=None, pair_index=None):
    try:
        return apply_task_rule_to_input(
            task_rule,
            input_grid,
            expected_grid=expected_grid,
            pair_index=pair_index,
        )
    except TypeError:
        try:
            return apply_task_rule_to_input(
                task_rule,
                input_grid,
            )
        except Exception:
            raise


def run_train_checks(task_id, task_rule, train_pairs):
    print_title("TRAIN APPLY CHECK")

    exact_count = 0

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        expected_grid = pair["output"]

        print_section(f"TRAIN PAIR {pair_index + 1}")

        try:
            predicted = apply_rule(
                task_rule,
                input_grid,
                expected_grid=expected_grid,
                pair_index=pair_index,
            )

            score, exact = score_prediction(predicted, expected_grid)

            if exact:
                exact_count += 1

            print(f"input shape    : {grid_shape(input_grid)}")
            print(f"expected shape : {grid_shape(expected_grid)}")
            print(f"predicted shape: {grid_shape(predicted)}")
            print(f"score          : {score}")
            print(f"exact          : {exact}")

            print_grid(input_grid, "INPUT")
            print_grid(expected_grid, "EXPECTED")

            if predicted is not None:
                print_grid(predicted, "PREDICTED")

                show_three_grids(
                    input_grid,
                    expected_grid,
                    predicted,
                    title_a=f"{task_id} TRAIN {pair_index + 1} INPUT",
                    title_b="EXPECTED",
                    title_c="PREDICTED",
                )
            else:
                print("PREDICTED: None")

        except Exception:
            print("ERROR APPLYING TRAIN PAIR")
            print(traceback.format_exc())

    print()
    print("=" * 80)
    print(f"TRAIN EXACT: {exact_count}/{len(train_pairs)}")
    print("=" * 80)


def run_test_checks(task_id, task_rule, test_pairs):
    print_title("TEST APPLY CHECK")

    for test_index, pair in enumerate(test_pairs):
        input_grid = pair["input"]

        print_section(f"TEST PAIR {test_index + 1}")

        try:
            predicted = apply_rule(
                task_rule,
                input_grid,
                expected_grid=None,
                pair_index=None,
            )

            print(f"input shape    : {grid_shape(input_grid)}")
            print(f"predicted shape: {grid_shape(predicted)}")

            print_grid(input_grid, "TEST INPUT")

            if predicted is not None:
                print_grid(predicted, "TEST PREDICTED")

                show_three_grids(
                    input_grid,
                    predicted,
                    predicted,
                    title_a=f"{task_id} TEST {test_index + 1} INPUT",
                    title_b="TEST PREDICTED",
                    title_c="TEST PREDICTED",
                )
            else:
                print("TEST PREDICTED: None")

        except Exception:
            print("ERROR APPLYING TEST PAIR")
            print(traceback.format_exc())


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) >= 2:
        task_id_or_path = sys.argv[1]
    else:
        task_id_or_path = input("Task id or json path: ").strip()

    task_id, task = load_task(task_id_or_path)

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    print_title(f"DEBUG COMPOSITION LAYOUT RULE DETAILS: {task_id}")

    router_result = choose_task_level_strategy(train_pairs)
    normalized = normalize_router_result(router_result)

    chosen_strategy = normalized["chosen_strategy"]
    task_rule = normalized["task_rule"]
    strategy_stats = normalized["strategy_stats"]

    print_section("ROUTER CHOICE")
    print(f"chosen strategy: {chosen_strategy}")

    print_rule_summary(task_rule)
    print_strategy_stats(strategy_stats)

    if not isinstance(task_rule, dict):
        print()
        print("Cannot apply rule because task_rule is not a dict.")
        return

    run_train_checks(task_id, task_rule, train_pairs)
    run_test_checks(task_id, task_rule, test_pairs)


if __name__ == "__main__":
    main()