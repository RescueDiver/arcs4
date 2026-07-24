# run_one_show_answer.py
import contextlib
import io
import json
import os

from OLD.arc_visualizer import show_three_grids
from core.grid_utils import print_grid

from reasoning.task_router import (
    choose_task_level_strategy,
    apply_task_rule_to_input,
    score_prediction,
)


QUIET_ENGINE_OUTPUT = True


def quiet_call(fn, *args, quiet=True, **kwargs):
    if not quiet:
        return fn(*args, **kwargs)

    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        return fn(*args, **kwargs)


def make_blank_like(grid):
    h = len(grid)
    w = len(grid[0]) if h else 0
    return [[0 for _ in range(w)] for _ in range(h)]


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def unwrap_task(raw, task_id):
    if "train" in raw:
        return raw

    if task_id in raw:
        return raw[task_id]

    if isinstance(raw, dict) and len(raw) == 1:
        only_key = next(iter(raw))
        return raw[only_key]

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

    raise FileNotFoundError(f"Could not find task: {value}")


def solve_with_task_rule(
    strategy_name,
    task_rule,
    input_grid,
    expected_grid=None,
    pair_index=None,
):
    return quiet_call(
        apply_task_rule_to_input,
        strategy_name=strategy_name,
        task_rule=task_rule,
        input_grid=input_grid,
        expected_grid=expected_grid,
        pair_index=pair_index,
        quiet=QUIET_ENGINE_OUTPUT,
    )


def print_router_result(router_result):
    chosen_strategy = router_result.get("best_strategy")
    task_rule = router_result.get("task_rule")
    strategy_stats = router_result.get("strategy_stats", {})

    print()
    print("TASK-LEVEL ROUTER")
    print("-" * 80)
    print(f"Chosen strategy: {chosen_strategy}")
    print(f"Has task rule  : {task_rule is not None}")

    print()
    print("Strategy stats:")
    for name, stats in sorted(strategy_stats.items()):
        print(
            f"  {name}: "
            f"train={stats.get('exact_count', 0)}/{stats.get('pair_count', 0)} "
            f"loo={stats.get('loo_exact_count', 0)}/{stats.get('loo_pair_count', 0)} "
            f"honest={stats.get('honest', False)} "
            f"adj={stats.get('total_adjusted_score', 0)}"
        )

    return chosen_strategy, task_rule


def show_train_pair(
    task_id,
    pair_index,
    chosen_strategy,
    task_rule,
    input_grid,
    expected_grid,
):
    predicted = solve_with_task_rule(
        strategy_name=chosen_strategy,
        task_rule=task_rule,
        input_grid=input_grid,
        expected_grid=expected_grid,
        pair_index=pair_index,
    )

    exact = predicted == expected_grid
    score = score_prediction(predicted, expected_grid) if predicted is not None else 0

    print()
    print("-" * 80)
    print(f"TRAIN PAIR {pair_index + 1}")
    print(f"Strategy: {chosen_strategy}")
    print(f"Score   : {score}")
    print(f"Exact   : {exact}")

    print_grid(input_grid, "INPUT")
    print_grid(expected_grid, "EXPECTED")

    if predicted is not None:
        print_grid(predicted, "PREDICTED")
        popup_predicted = predicted
    else:
        print("PREDICTED: None")
        popup_predicted = make_blank_like(expected_grid)

    show_three_grids(
        input_grid,
        expected_grid,
        popup_predicted,
        title_a=f"{task_id} TRAIN {pair_index + 1} INPUT",
        title_b="EXPECTED",
        title_c="PREDICTED",
    )

    return exact


def show_test_pair(
    task_id,
    test_index,
    chosen_strategy,
    task_rule,
    input_grid,
    expected_grid=None,
):
    predicted = solve_with_task_rule(
        strategy_name=chosen_strategy,
        task_rule=task_rule,
        input_grid=input_grid,
        expected_grid=None,
        pair_index=None,
    )

    print()
    print("-" * 80)
    print(f"TEST PAIR {test_index + 1}")
    print(f"Strategy: {chosen_strategy}")

    print_grid(input_grid, "TEST INPUT")

    if expected_grid is not None:
        exact = predicted == expected_grid
        score = score_prediction(predicted, expected_grid) if predicted is not None else 0

        print(f"Score   : {score}")
        print(f"Exact   : {exact}")
        print_grid(expected_grid, "TEST EXPECTED")

    if predicted is not None:
        print_grid(predicted, "TEST PREDICTED")
    else:
        print("TEST PREDICTED: None")

    if expected_grid is not None:
        popup_predicted = predicted if predicted is not None else make_blank_like(expected_grid)

        show_three_grids(
            input_grid,
            expected_grid,
            popup_predicted,
            title_a=f"{task_id} TEST {test_index + 1} INPUT",
            title_b="EXPECTED",
            title_c="PREDICTED",
        )
    else:
        if predicted is not None:
            show_three_grids(
                input_grid,
                predicted,
                predicted,
                title_a=f"{task_id} TEST {test_index + 1} INPUT",
                title_b="PREDICTED",
                title_c="PREDICTED",
            )
        else:
            print("No test prediction to show in popup.")


def main():
    task_id_or_path = input("Task id or json path: ").strip()
    task_id, task = load_task(task_id_or_path)

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    print()
    print("=" * 80)
    print(f"TASK: {task_id}")
    print("=" * 80)

    router_result = quiet_call(
        choose_task_level_strategy,
        train_pairs,
        debug=False,
        quiet=QUIET_ENGINE_OUTPUT,
    )

    chosen_strategy, task_rule = print_router_result(router_result)

    print()
    print("=" * 80)
    print("TRAIN PAIR ANSWERS")
    print("=" * 80)

    train_right = 0
    train_wrong = 0

    for pair_index, pair in enumerate(train_pairs):
        exact = show_train_pair(
            task_id=task_id,
            pair_index=pair_index,
            chosen_strategy=chosen_strategy,
            task_rule=task_rule,
            input_grid=pair["input"],
            expected_grid=pair["output"],
        )

        if exact:
            train_right += 1
        else:
            train_wrong += 1

    print()
    print("=" * 80)
    print("TEST PAIR ANSWERS")
    print("=" * 80)

    for test_index, pair in enumerate(test_pairs):
        show_test_pair(
            task_id=task_id,
            test_index=test_index,
            chosen_strategy=chosen_strategy,
            task_rule=task_rule,
            input_grid=pair["input"],
            expected_grid=pair.get("output"),
        )

    print()
    print("TASK RULE DEBUG")
    print("-" * 80)

    if isinstance(task_rule, dict):
        print(f"Rule family: {task_rule.get('family')}")
        print(f"Rule type  : {task_rule.get('rule_type')}")
        print(f"Keys       : {sorted(task_rule.keys())}")
    else:
        print(f"Task rule type: {type(task_rule)}")


    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Task: {task_id}")
    print(f"Chosen strategy: {chosen_strategy}")
    print(f"Train Right: {train_right}")
    print(f"Train Wrong: {train_wrong}")
    print("=" * 80)


if __name__ == "__main__":
    main()