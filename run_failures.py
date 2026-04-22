import json
import os

from arc_visualizer import show_three_grids
from core.grid_utils import print_grid
from reasoning.task_router import solve_pair_with_multiple_strategies
from debug.debug_utils import debug_strategy_scores, debug_router_adjustments

def load_task_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def should_show_case(exact, shown_wrong, max_wrong):
    return (not exact) and shown_wrong < max_wrong


def run_failures():
    base_dir = os.path.dirname(__file__)
    failures_dir = os.path.join(base_dir, "data_failures", "extracted_tasks")

    if not os.path.isdir(failures_dir):
        print(f"Missing folder: {failures_dir}")
        return

    task_files = sorted(
        f for f in os.listdir(failures_dir)
        if f.endswith(".json")
    )

    if not task_files:
        print("No extracted failure task files found.")
        return

    # -----------------------------
    # DISPLAY SETTINGS
    # -----------------------------
    max_wrong_visuals_total = 3
    shown_wrong_visuals = 0

    # -----------------------------
    # OVERALL STATS
    # -----------------------------
    total_pairs = 0
    total_right = 0
    total_wrong = 0
    overall_strategy_counts = {}

    fully_solved_tasks = 0
    partially_solved_tasks = 0
    failed_tasks = 0

    for file_name in task_files:
        file_path = os.path.join(failures_dir, file_name)
        tasks = load_task_file(file_path)

        for task_id, task in tasks.items():
            print("\n" + "=" * 60)
            print(f"FAILURE TASK: {task_id}")
            print("=" * 60)

            task_total = 0
            task_right = 0
            task_wrong = 0
            task_strategy_counts = {}

            for idx, pair in enumerate(task.get("train", []), start=1):
                input_grid = pair["input"]
                output_grid = pair["output"]

                result = solve_pair_with_multiple_strategies(input_grid, output_grid)

                task_total += 1
                total_pairs += 1

                print(f"\n--- TRAIN PAIR {idx} ---")

                if result is None:
                    print("No result")
                    task_wrong += 1
                    total_wrong += 1
                    continue

                exact = result.get("exact", False)
                strategy = result.get("strategy", "unknown")
                score = result.get("score", None)
                selector = result.get("selector", None)
                transform = result.get("transform", None)

                task_strategy_counts[strategy] = task_strategy_counts.get(strategy, 0) + 1
                overall_strategy_counts[strategy] = overall_strategy_counts.get(strategy, 0) + 1

                if exact:
                    task_right += 1
                    total_right += 1
                else:
                    task_wrong += 1
                    total_wrong += 1

                print(f"Strategy: {strategy}")
                print(f"Score: {score}")
                print(f"Exact: {exact}")

                if selector is not None:
                    print(f"Selector: {selector}")

                if transform is not None:
                    print(f"Transform: {transform}")

                if should_show_case(exact, shown_wrong_visuals, max_wrong_visuals_total):
                    shown_wrong_visuals += 1

                    print_grid(input_grid, "INPUT")
                    print_grid(output_grid, "EXPECTED")

                    if "object" in result and result["object"] is not None:
                        print_grid(result["object"]["patch"], "SELECTED OBJECT")

                    print_grid(result["predicted"], "PREDICTED")
                    show_three_grids(input_grid, result["predicted"], output_grid)

            if task_total > 0:
                if task_right == task_total:
                    fully_solved_tasks += 1
                elif task_right == 0:
                    failed_tasks += 1
                else:
                    partially_solved_tasks += 1

            right_pct = (task_right / task_total * 100) if task_total else 0.0
            wrong_pct = (task_wrong / task_total * 100) if task_total else 0.0

            print("\n" + "-" * 60)
            print(f"TASK SUMMARY: {task_id}")
            print(f"Right: {task_right}")
            print(f"Wrong: {task_wrong}")
            print(f"Percent Right: {right_pct:.2f}%")
            print(f"Percent Wrong: {wrong_pct:.2f}%")
            print("Strategy Wins:")
            for name, count in sorted(task_strategy_counts.items()):
                print(f"  {name}: {count}")
            print("-" * 60)

    total_right_pct = (total_right / total_pairs * 100) if total_pairs else 0.0
    total_wrong_pct = (total_wrong / total_pairs * 100) if total_pairs else 0.0

    print("\n" + "=" * 60)
    print("FAILURE-ONLY FINAL SUMMARY")
    print("=" * 60)
    print(f"Task files checked: {len(task_files)}")
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
    print("=" * 60)


if __name__ == "__main__":
    run_failures()