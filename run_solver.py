import os
import json

from reasoning.task_router import solve_pair_with_multiple_strategies
from reasoning.pattern_rule_engine import learn_pattern_rule_from_train_pairs

from debug.debug_utils import print_grid, show_three_grids


def load_tasks(filename):
    base_dir = os.path.join(os.path.dirname(__file__), "data")

    # CASE 1: run every .json file in data/
    if filename.lower() == "data":
        tasks = []

        for file in os.listdir(base_dir):
            if not file.endswith(".json"):
                continue

            path = os.path.join(base_dir, file)

            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            # Standard single ARC task file
            if isinstance(raw, dict) and ("train" in raw or "test" in raw):
                task_id = file.replace(".json", "")
                tasks.append((task_id, raw))

            # Multi-task file
            elif isinstance(raw, dict):
                for k, v in raw.items():
                    if isinstance(v, dict) and ("train" in v or "test" in v):
                        tasks.append((k, v))

        return tasks

    # CASE 2: run one file
    if not filename.endswith(".json"):
        filename += ".json"

    path = os.path.join(base_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Task file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and ("train" in raw or "test" in raw):
        task_id = filename.replace(".json", "")
        return [(task_id, raw)]

    if isinstance(raw, dict):
        tasks = []
        for k, v in raw.items():
            if isinstance(v, dict) and ("train" in v or "test" in v):
                tasks.append((k, v))
        return tasks

    raise ValueError(f"Unexpected task format in {path}")


def should_show_case(exact, shown_wrong, max_wrong, shown_correct, max_correct, show_correct):
    if not exact and shown_wrong < max_wrong:
        return True
    if exact and show_correct and shown_correct < max_correct:
        return True
    return False


def save_wrong_cases_txt(base_dir, wrong_cases):
    output_path = os.path.join(base_dir, "wrong_cases.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WRONG TRAIN PAIRS\n")
        f.write("=" * 60 + "\n\n")

        for i, case in enumerate(wrong_cases, start=1):
            f.write(f"{i}. TASK: {case['task_id']} | PAIR: {case['pair_index']}\n")
            f.write(f"   Strategy : {case['strategy']}\n")
            f.write(f"   Score    : {case['score']}\n")
            f.write(f"   Selector : {case['selector']}\n")
            f.write(f"   Transform: {case['transform']}\n")
            f.write("\n")

    return output_path


def save_wrong_tasks_only(base_dir, wrong_cases):
    output_path = os.path.join(base_dir, "wrong_tasks_only.txt")

    unique_task_ids = sorted({case["task_id"] for case in wrong_cases})

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("TASKS WITH AT LEAST ONE WRONG TRAIN PAIR\n")
        f.write("=" * 60 + "\n\n")

        for task_id in unique_task_ids:
            f.write(task_id + "\n")

    return output_path


def save_wrong_cases_json(base_dir, wrong_cases):
    output_path = os.path.join(base_dir, "wrong_cases.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(wrong_cases, f, indent=2)

    return output_path


def run():
    base_dir = os.path.dirname(__file__)

    filename = input("Enter task file (e.g. fc7.json, 135.json, or data): ").strip()
    tasks = load_tasks(filename)

    # DISPLAY SETTINGS
    show_correct_visuals = False
    max_correct_visuals = 0
    max_wrong_visuals_total = 3

    shown_wrong_visuals = 0
    shown_correct_visuals = 0

    # OVERALL STATS
    total_pairs = 0
    total_right = 0
    total_wrong = 0
    overall_strategy_counts = {}

    fully_solved_tasks = 0
    partially_solved_tasks = 0
    failed_tasks = 0

    wrong_cases = []

    for task_id, task in tasks:
        print("\n" + "=" * 60)
        print(f"TASK: {task_id}")
        print("=" * 60)

        train_pairs = task.get("train", [])
        task_total = 0
        task_right = 0
        task_wrong = 0
        task_strategy_counts = {}

        for idx, pair in enumerate(train_pairs, start=1):
            input_grid = pair["input"]
            output_grid = pair["output"]


            task_total += 1
            total_pairs += 1

            print(f"\n--- TRAIN PAIR {idx} ---")

            # 🔥 THIS LINE WAS MISSING
            result = solve_pair_with_multiple_strategies(input_grid, output_grid)

            if result is None:
                print("No result")
                task_wrong += 1
                total_wrong += 1

                wrong_cases.append({
                    "task_id": task_id,
                    "pair_index": idx,
                    "strategy": "no_result",
                    "score": None,
                    "selector": None,
                    "transform": None,
                })
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

                wrong_cases.append({
                    "task_id": task_id,
                    "pair_index": idx,
                    "strategy": strategy,
                    "score": score,
                    "selector": selector,
                    "transform": transform,
                })

            print(f"Strategy: {strategy}")
            print(f"Score: {score}")
            print(f"Exact: {exact}")

            if selector is not None:
                print(f"Selector: {selector}")

            if transform is not None:
                print(f"Transform: {transform}")

            show_case = should_show_case(
                exact=exact,
                shown_wrong=shown_wrong_visuals,
                max_wrong=max_wrong_visuals_total,
                shown_correct=shown_correct_visuals,
                max_correct=max_correct_visuals,
                show_correct=show_correct_visuals,
            )

            if show_case:
                if exact:
                    shown_correct_visuals += 1
                else:
                    shown_wrong_visuals += 1

                print_grid(input_grid, "INPUT")
                print_grid(output_grid, "EXPECTED")

                if "object" in result and result["object"] is not None:
                    print_grid(result["object"]["patch"], "SELECTED OBJECT")

                print_grid(result.get("predicted"), "PREDICTED")

                try:
                    show_three_grids(input_grid, result.get("predicted"), output_grid)
                except Exception as e:
                    print(f"[VISUAL ERROR] {e}")

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

    wrong_cases_txt_path = save_wrong_cases_txt(base_dir, wrong_cases)
    wrong_tasks_only_path = save_wrong_tasks_only(base_dir, wrong_cases)
    wrong_cases_json_path = save_wrong_cases_json(base_dir, wrong_cases)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
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
    print(f"Wrong-case list saved to : {wrong_cases_txt_path}")
    print(f"Wrong-task list saved to : {wrong_tasks_only_path}")
    print(f"Wrong-case JSON saved to : {wrong_cases_json_path}")
    print("=" * 60)