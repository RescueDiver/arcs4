# run_failures.py
import contextlib
import io
import json
import os

from OLD.arc_visualizer import show_three_grids
from core.grid_utils import print_grid

from reasoning.task_router import (
    choose_task_level_strategy,
    solve_pair_with_multiple_strategies,
    apply_task_rule_to_input,
    score_prediction,
)


# ============================================================
# SETTINGS
# ============================================================

QUIET_ENGINE_OUTPUT = True
PRINT_STRATEGY_STATS = True
PRINT_SOLVED_TASKS = True
PRINT_ONLY_WRONG_PAIRS = True
SHOW_WRONG_VISUALS = False
MAX_WRONG_VISUALS_TOTAL = 0
MAX_TASK_FILE_BYTES = 200_000

# ============================================================
# FILE LOADING
# ============================================================

def load_task_file(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# QUIET CALL WRAPPER
# ============================================================

def quiet_call(fn, *args, quiet=True, **kwargs):
    """
    Run noisy solver functions without letting their debug prints flood console.
    """
    if not quiet:
        return fn(*args, **kwargs)

    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        return fn(*args, **kwargs)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def should_show_case(exact, shown_wrong, max_wrong):
    if not SHOW_WRONG_VISUALS:
        return False

    return (not exact) and shown_wrong < max_wrong


def print_strategy_stats(strategy_stats):
    if not PRINT_STRATEGY_STATS:
        return

    if not strategy_stats:
        print("No strategy stats.")
        return

    for name, stats in sorted(strategy_stats.items()):
        exact = stats.get("exact_count", 0)
        pairs = stats.get("pair_count", 0)
        loo_exact = stats.get("loo_exact_count", 0)
        loo_pairs = stats.get("loo_pair_count", 0)
        total_adj = stats.get("total_adjusted_score", 0)
        honest = stats.get("honest", False)

        print(
            f"  {name}: "
            f"train={exact}/{pairs} "
            f"loo={loo_exact}/{loo_pairs} "
            f"honest={honest} "
            f"adj={total_adj}"
        )


def record_strategy_win(strategy_counts, strategy_name):
    if strategy_name is None:
        strategy_name = "unknown"

    strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1


def print_task_header(task_id):
    print()
    print("=" * 60)
    print(f"FAILURE TASK: {task_id}")
    print("=" * 60)


def print_router_summary(chosen_strategy, task_rule, strategy_stats):
    print()
    print("TASK-LEVEL ROUTER")
    print("-" * 60)
    print(f"Chosen strategy: {chosen_strategy}")
    print(f"Has task rule  : {task_rule is not None}")

    if PRINT_STRATEGY_STATS:
        print("Strategy stats:")
        print_strategy_stats(strategy_stats)


def print_task_summary(task_id, task_right, task_wrong, task_strategy_counts):
    task_total = task_right + task_wrong
    right_pct = (task_right / task_total * 100) if task_total else 0.0
    wrong_pct = (task_wrong / task_total * 100) if task_total else 0.0

    print()
    print("-" * 60)
    print(f"TASK SUMMARY: {task_id}")
    print(f"Right: {task_right}")
    print(f"Wrong: {task_wrong}")
    print(f"Percent Right: {right_pct:.2f}%")
    print(f"Percent Wrong: {wrong_pct:.2f}%")

    if task_strategy_counts:
        print("Strategy Wins:")
        for name, count in sorted(task_strategy_counts.items()):
            print(f"  {name}: {count}")

    print("-" * 60)


# ============================================================
# TASK-LEVEL TRAIN PAIR SOLVING
# ============================================================

def solve_train_pair_with_task_level_strategy(
    strategy_name,
    task_rule,
    input_grid,
    output_grid,
    pair_index,
):
    """
    Apply the chosen task-level strategy to a train pair.
    """
    predicted = quiet_call(
        apply_task_rule_to_input,
        strategy_name=strategy_name,
        task_rule=task_rule,
        input_grid=input_grid,
        expected_grid=output_grid,
        pair_index=pair_index,
        quiet=QUIET_ENGINE_OUTPUT,
    )

    if predicted is None:
        return {
            "strategy": strategy_name,
            "predicted": None,
            "score": 0,
            "exact": False,
        }

    score = score_prediction(predicted, output_grid)
    exact = predicted == output_grid

    return {
        "strategy": strategy_name,
        "predicted": predicted,
        "score": score,
        "exact": exact,
    }


def solve_train_pair(
    strategy_name,
    task_rule,
    input_grid,
    output_grid,
    pair_index,
):
    """
    Use the chosen task-level router strategy when one exists.
    """
    task_level_families = {
        "visual_symbolic_family",
        "pattern_canvas_family",
        "region_object_family",
        "composition_layout_family",
        "anchor_repair_family",
    }

    if strategy_name in task_level_families and task_rule is not None:
        return solve_train_pair_with_task_level_strategy(
            strategy_name=strategy_name,
            task_rule=task_rule,
            input_grid=input_grid,
            output_grid=output_grid,
            pair_index=pair_index,
        )

    result = quiet_call(
        solve_pair_with_multiple_strategies,
        input_grid,
        output_grid,
        debug=False,
        quiet=QUIET_ENGINE_OUTPUT,
    )

    return result


# ============================================================
# MAIN FAILURE RUNNER
# ============================================================

def run_failures():
    base_dir = os.path.dirname(__file__)
    failures_dir = os.path.join(base_dir, "data_failures", "extracted_tasks")

    if not os.path.isdir(failures_dir):
        print(f"Missing folder: {failures_dir}")
        return

    task_files = sorted(
        file_name
        for file_name in os.listdir(failures_dir)
        if file_name.endswith(".json")
    )

    if not task_files:
        print("No extracted failure task files found.")
        return

    shown_wrong_visuals = 0

    total_pairs = 0
    total_right = 0
    total_wrong = 0

    overall_strategy_counts = {}
    task_level_strategy_counts = {}

    fully_solved_tasks = 0
    partially_solved_tasks = 0
    failed_tasks = 0
    wrong_task_ids = []

    print()
    print("=" * 60)
    print("FAILURE-ONLY RUN")
    print("=" * 60)
    print(f"Task files found: {len(task_files)}")
    print(f"Quiet engine output: {QUIET_ENGINE_OUTPUT}")
    print("=" * 60)

    skipped_large_files = []

    for file_name in task_files:
        file_path = os.path.join(failures_dir, file_name)

        file_size = os.path.getsize(file_path)

        if file_size > MAX_TASK_FILE_BYTES:
            task_id = file_name.replace(".json", "")
            skipped_large_files.append((task_id, file_size))

            print()
            print("=" * 60)
            print(f"SKIPPED LARGE TASK FILE: {task_id}")
            print("=" * 60)
            print(f"File size: {file_size} bytes")
            print(f"Limit    : {MAX_TASK_FILE_BYTES} bytes")
            continue

        tasks = load_task_file(file_path)

        for task_id, task in tasks.items():
            train_pairs = task.get("train", [])

            print_task_header(task_id)

            task_router_result = quiet_call(
                choose_task_level_strategy,
                train_pairs,
                debug=False,
                quiet=QUIET_ENGINE_OUTPUT,
            )

            chosen_strategy = task_router_result.get("best_strategy")
            task_rule = task_router_result.get("task_rule")
            strategy_stats = task_router_result.get("strategy_stats", {})

            print_router_summary(
                chosen_strategy,
                task_rule,
                strategy_stats,
            )

            record_strategy_win(
                task_level_strategy_counts,
                chosen_strategy,
            )

            task_total = 0
            task_right = 0
            task_wrong = 0
            task_strategy_counts = {}

            for pair_index, pair in enumerate(train_pairs):
                input_grid = pair["input"]
                output_grid = pair["output"]

                result = solve_train_pair(
                    strategy_name=chosen_strategy,
                    task_rule=task_rule,
                    input_grid=input_grid,
                    output_grid=output_grid,
                    pair_index=pair_index,
                )

                task_total += 1
                total_pairs += 1

                if result is None:
                    task_wrong += 1
                    total_wrong += 1

                    print()
                    print(f"--- TRAIN PAIR {pair_index + 1} ---")
                    print("No result")
                    continue

                exact = result.get("exact", False)
                strategy = result.get("strategy", "unknown")
                score = result.get("score", None)

                record_strategy_win(task_strategy_counts, strategy)
                record_strategy_win(overall_strategy_counts, strategy)

                if exact:
                    task_right += 1
                    total_right += 1
                else:
                    task_wrong += 1
                    total_wrong += 1

                    print()
                    print(f"--- WRONG TRAIN PAIR {pair_index + 1} ---")
                    print(f"Strategy: {strategy}")
                    print(f"Score   : {score}")
                    print(f"Exact   : {exact}")

                    selector = result.get("selector")
                    transform = result.get("transform")

                    if selector is not None:
                        print(f"Selector: {selector}")

                    if transform is not None:
                        print(f"Transform: {transform}")

                predicted = result.get("predicted")

                if should_show_case(
                    exact,
                    shown_wrong_visuals,
                    MAX_WRONG_VISUALS_TOTAL,
                ):
                    shown_wrong_visuals += 1

                    print_grid(input_grid, "INPUT")
                    print_grid(output_grid, "EXPECTED")

                    if result.get("object") is not None:
                        print_grid(result["object"]["patch"], "SELECTED OBJECT")

                    if predicted is not None:
                        print_grid(predicted, "PREDICTED")
                        show_three_grids(input_grid, predicted, output_grid)

            if task_total > 0:
                if task_right == task_total:
                    fully_solved_tasks += 1
                elif task_right == 0:
                    failed_tasks += 1
                    wrong_task_ids.append(task_id)
                else:
                    partially_solved_tasks += 1
                    wrong_task_ids.append(task_id)

            if PRINT_SOLVED_TASKS or task_wrong > 0:
                print_task_summary(
                    task_id,
                    task_right,
                    task_wrong,
                    task_strategy_counts,
                )

    total_right_pct = (total_right / total_pairs * 100) if total_pairs else 0.0
    total_wrong_pct = (total_wrong / total_pairs * 100) if total_pairs else 0.0

    print()
    print("Skipped Large Files:")
    if skipped_large_files:
        for task_id, file_size in skipped_large_files:
            print(f"  {task_id}: {file_size} bytes")
    else:
        print("  None")
    print()
    print("=" * 60)
    print("FAILURE-ONLY FINAL SUMMARY")
    print("=" * 60)
    print(f"Task files checked: {len(task_files)}")
    print(f"Total Pairs: {total_pairs}")
    print(f"Total Right: {total_right}")
    print(f"Total Wrong: {total_wrong}")
    print(f"Percent Right: {total_right_pct:.2f}%")
    print(f"Percent Wrong: {total_wrong_pct:.2f}%")

    print()
    print(f"Tasks Fully Solved: {fully_solved_tasks}")
    print(f"Tasks Partially Solved: {partially_solved_tasks}")
    print(f"Tasks Failed: {failed_tasks}")

    print()
    print("Overall Strategy Wins:")
    for name, count in sorted(overall_strategy_counts.items()):
        print(f"  {name}: {count}")

    print()
    print("Task-Level Strategy Choices:")
    for name, count in sorted(task_level_strategy_counts.items()):
        print(f"  {name}: {count}")

    print()
    print("Wrong / Partial Task IDs:")
    if wrong_task_ids:
        for task_id in wrong_task_ids:
            print(f"  {task_id}")
    else:
        print("  None")

    print("=" * 60)


if __name__ == "__main__":
    run_failures()