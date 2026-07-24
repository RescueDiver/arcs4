"""
Honest leave-one-out benchmark for ARCs4.

This file does not change the solver.  It measures how the current task-level
router behaves when one training pair is treated exactly like a Kaggle test
pair:

    1. Remove one training pair completely.
    2. Learn and choose a rule from the remaining pairs.
    3. Predict from the hidden input with no expected output and no pair index.
    4. Compare with the hidden output only after the prediction is complete.

Run the complete public set:

    python run_honest_benchmark.py

Run one task:

    python run_honest_benchmark.py --task 136b0064

Run a quick smoke test:

    python run_honest_benchmark.py --limit 3
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any

from reasoning.task_router import (
    apply_task_rule_to_input,
    choose_task_level_strategy,
)


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(PROJECT_DIR, "data", "data.json")
DEFAULT_JSON_REPORT = os.path.join(PROJECT_DIR, "honest_benchmark_results.json")
DEFAULT_TEXT_REPORT = os.path.join(PROJECT_DIR, "honest_benchmark_summary.txt")
BENCHMARK_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a leak-free leave-one-out benchmark on ARCs4.",
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help="ARC JSON file to load (default: data/data.json).",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Run only this task id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N selected tasks.",
    )
    parser.add_argument(
        "--verbose-router",
        action="store_true",
        help="Show output printed by the existing router.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of task processes to run in parallel (default: 1).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse compatible completed tasks already in the JSON report.",
    )
    parser.add_argument(
        "--json-report",
        default=DEFAULT_JSON_REPORT,
        help="Path for the detailed JSON report.",
    )
    parser.add_argument(
        "--text-report",
        default=DEFAULT_TEXT_REPORT,
        help="Path for the readable text summary.",
    )
    return parser.parse_args()


def solver_fingerprint() -> str:
    """Identify the router version so resume never mixes different solvers."""
    router_path = os.path.join(PROJECT_DIR, "reasoning", "task_router.py")
    digest = hashlib.sha256()

    with open(router_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_compatible_checkpoint(
    json_path: str,
    fingerprint: str,
) -> tuple[dict[str, dict[str, Any]], float]:
    if not os.path.exists(json_path):
        return {}, 0.0

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, ValueError, TypeError):
        return {}, 0.0

    summary = report.get("summary") or {}

    if summary.get("benchmark_version") != BENCHMARK_VERSION:
        return {}, 0.0

    if summary.get("solver_fingerprint") != fingerprint:
        return {}, 0.0

    completed = {}

    for task_result in report.get("tasks") or []:
        task_id = task_result.get("task_id")
        if task_id:
            completed[str(task_id)] = task_result

    previous_elapsed = summary.get("elapsed_seconds", 0.0)
    if not isinstance(previous_elapsed, (int, float)):
        previous_elapsed = 0.0

    return completed, float(previous_elapsed)


def load_tasks(path: str) -> list[tuple[str, dict[str, Any]]]:
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_DIR, path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Task file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in: {path}")

    if "train" in raw or "test" in raw:
        task_id = os.path.splitext(os.path.basename(path))[0]
        return [(task_id, raw)]

    tasks = []

    for task_id, task in raw.items():
        if isinstance(task, dict) and ("train" in task or "test" in task):
            tasks.append((str(task_id), task))

    if not tasks:
        raise ValueError(f"No ARC tasks found in: {path}")

    return tasks


def quiet_call(function, *args, quiet: bool, **kwargs):
    if not quiet:
        return function(*args, **kwargs)

    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            return function(*args, **kwargs)


def grid_shape(grid: Any) -> tuple[int, int]:
    if not isinstance(grid, list) or not grid:
        return 0, 0

    first_row = grid[0]
    width = len(first_row) if isinstance(first_row, list) else 0
    return len(grid), width




def choose_rule(train_pairs: list[dict[str, Any]], quiet: bool) -> dict[str, Any]:
    choice = quiet_call(
        choose_task_level_strategy,
        train_pairs,
        debug=False,
        quiet=quiet,
    )

    if not isinstance(choice, dict):
        return {
            "best_strategy": None,
            "task_rule": None,
            "rule": None,
            "strategy_stats": {},
        }

    return choice


def selected_replay_stats(
    choice: dict[str, Any],
    pair_count: int,
) -> tuple[int, int]:
    """Read the router's existing full-training reconstruction score."""
    strategy = choice.get("best_strategy")
    strategy_stats = choice.get("strategy_stats") or {}
    stats = strategy_stats.get(strategy) or {}

    replay_total = stats.get("pair_count")
    replay_exact = stats.get("exact_count")

    if not isinstance(replay_total, int):
        replay_total = pair_count

    if not isinstance(replay_exact, int):
        replay_exact = 0

    return replay_exact, replay_total


def predict_hidden_pair(
    train_pairs: list[dict[str, Any]],
    hidden_index: int,
    quiet: bool,
) -> dict[str, Any]:
    visible_pairs = [
        copy.deepcopy(pair)
        for index, pair in enumerate(train_pairs)
        if index != hidden_index
    ]

    hidden_pair = train_pairs[hidden_index]
    hidden_input = copy.deepcopy(hidden_pair.get("input"))
    hidden_expected = copy.deepcopy(hidden_pair.get("output"))

    started = time.perf_counter()
    error = None

    try:
        choice = choose_rule(visible_pairs, quiet=quiet)
        strategy = choice.get("best_strategy")
        task_rule = choice.get("task_rule") or choice.get("rule")

        # CRITICAL HONESTY BOUNDARY:
        # The hidden output and its original pair number are not passed.
        predicted = quiet_call(
            apply_task_rule_to_input,
            strategy_name=strategy,
            task_rule=task_rule,
            input_grid=hidden_input,
            expected_grid=None,
            pair_index=None,
            quiet=quiet,
        )
    except Exception as exc:  # keep the full benchmark running
        choice = {}
        strategy = None
        task_rule = None
        predicted = None
        error = f"{type(exc).__name__}: {exc}"

    elapsed_seconds = time.perf_counter() - started
    predicted_shape = grid_shape(predicted)
    expected_shape = grid_shape(hidden_expected)

    return {
        "hidden_pair": hidden_index + 1,
        "visible_pair_count": len(visible_pairs),
        "strategy": strategy,
        "rule_type": task_rule.get("rule_type") if isinstance(task_rule, dict) else None,
        "chosen_inner_strategy": (
            task_rule.get("chosen_inner_strategy")
            if isinstance(task_rule, dict)
            else None
        ),
        "predicted": predicted,
        "expected": hidden_expected,
        "exact": predicted == hidden_expected,
        "prediction_missing": predicted is None,
        "shape_exact": predicted_shape == expected_shape,
        "predicted_shape": list(predicted_shape),
        "expected_shape": list(expected_shape),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "error": error,
    }


def benchmark_task(
    task_id: str,
    task: dict[str, Any],
    quiet: bool,
) -> dict[str, Any]:
    train_pairs = task.get("train") or []
    pair_count = len(train_pairs)

    if pair_count < 2:
        return {
            "task_id": task_id,
            "train_pair_count": pair_count,
            "status": "not_testable",
            "reason": "Honest leave-one-out requires at least two training pairs.",
            "replay_strategy": None,
            "replay_exact": 0,
            "replay_total": pair_count,
            "honest_exact": 0,
            "honest_total": 0,
            "folds": [],
        }

    replay_choice = choose_rule(copy.deepcopy(train_pairs), quiet=quiet)
    replay_strategy = replay_choice.get("best_strategy")
    replay_exact, replay_total = selected_replay_stats(
        replay_choice,
        pair_count,
    )

    folds = [
        predict_hidden_pair(
            train_pairs=train_pairs,
            hidden_index=hidden_index,
            quiet=quiet,
        )
        for hidden_index in range(pair_count)
    ]

    honest_exact = sum(1 for fold in folds if fold["exact"])

    if honest_exact == pair_count:
        status = "fully_solved"
    elif honest_exact == 0:
        status = "failed"
    else:
        status = "partially_solved"

    return {
        "task_id": task_id,
        "train_pair_count": pair_count,
        "status": status,
        "replay_strategy": replay_strategy,
        "replay_exact": replay_exact,
        "replay_total": replay_total,
        "honest_exact": honest_exact,
        "honest_total": pair_count,
        "folds": folds,
    }


def benchmark_task_job(
    task_id: str,
    task: dict[str, Any],
    quiet: bool,
) -> dict[str, Any]:
    """Top-level worker entry point; required for Windows multiprocessing."""
    return benchmark_task(task_id=task_id, task=task, quiet=quiet)


def percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator * 100.0


def build_summary(
    data_path: str,
    task_results: list[dict[str, Any]],
    elapsed_seconds: float,
    fingerprint: str,
) -> dict[str, Any]:
    testable = [item for item in task_results if item["status"] != "not_testable"]
    folds = [fold for item in testable for fold in item["folds"]]

    replay_exact = sum(item["replay_exact"] for item in testable)
    replay_total = sum(item["replay_total"] for item in testable)
    honest_exact = sum(item["honest_exact"] for item in testable)
    honest_total = sum(item["honest_total"] for item in testable)

    family_totals: dict[str, int] = defaultdict(int)
    family_exact: dict[str, int] = defaultdict(int)

    for fold in folds:
        family = fold.get("strategy") or "no_strategy"
        family_totals[family] += 1
        if fold.get("exact"):
            family_exact[family] += 1

    family_summary = []

    for family in sorted(family_totals):
        total = family_totals[family]
        exact = family_exact[family]
        family_summary.append(
            {
                "family": family,
                "exact": exact,
                "total": total,
                "percent": round(percent(exact, total), 4),
            }
        )

    status_counts = Counter(item["status"] for item in task_results)

    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "benchmark_version": BENCHMARK_VERSION,
        "solver_fingerprint": fingerprint,
        "data_path": os.path.abspath(data_path),
        "honesty_contract": {
            "hidden_output_passed_to_solver": False,
            "hidden_pair_index_passed_to_solver": False,
            "pair_level_expected_output_fallback_allowed": False,
            "comparison_occurs_after_prediction": True,
        },
        "task_count": len(task_results),
        "testable_task_count": len(testable),
        "not_testable_task_count": status_counts["not_testable"],
        "fully_solved_tasks": status_counts["fully_solved"],
        "partially_solved_tasks": status_counts["partially_solved"],
        "failed_tasks": status_counts["failed"],
        "train_replay_exact": replay_exact,
        "train_replay_total": replay_total,
        "train_replay_percent": round(percent(replay_exact, replay_total), 4),
        "honest_exact": honest_exact,
        "honest_total": honest_total,
        "honest_percent": round(percent(honest_exact, honest_total), 4),
        "shape_exact": sum(1 for fold in folds if fold["shape_exact"]),
        "prediction_missing": sum(1 for fold in folds if fold["prediction_missing"]),
        "errors": sum(1 for fold in folds if fold["error"] is not None),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "family_summary": family_summary,
    }


def render_summary(summary: dict[str, Any]) -> str:
    lines = [
        "HONEST ARCs4 LEAVE-ONE-OUT BENCHMARK",
        "=" * 72,
        f"Data file             : {summary['data_path']}",
        f"Tasks                 : {summary['task_count']}",
        f"Testable tasks        : {summary['testable_task_count']}",
        f"Not testable          : {summary['not_testable_task_count']}",
        "",
        "TRAIN REPLAY (NOT HONEST)",
        "-" * 72,
        (
            f"Exact grids           : {summary['train_replay_exact']} / "
            f"{summary['train_replay_total']}"
        ),
        f"Replay percent        : {summary['train_replay_percent']:.2f}%",
        "",
        "HIDDEN-PAIR RESULT (HONEST)",
        "-" * 72,
        f"Exact grids           : {summary['honest_exact']} / {summary['honest_total']}",
        f"Honest percent        : {summary['honest_percent']:.2f}%",
        f"Correct output shape  : {summary['shape_exact']} / {summary['honest_total']}",
        f"Missing predictions   : {summary['prediction_missing']}",
        f"Errors                : {summary['errors']}",
        "",
        "TASK RESULTS",
        "-" * 72,
        f"Fully solved          : {summary['fully_solved_tasks']}",
        f"Partially solved      : {summary['partially_solved_tasks']}",
        f"Failed                : {summary['failed_tasks']}",
        "",
        "HONEST RESULT BY CHOSEN FAMILY",
        "-" * 72,
    ]

    if summary["family_summary"]:
        for item in summary["family_summary"]:
            lines.append(
                f"{item['family']:<30} "
                f"{item['exact']:>4}/{item['total']:<4} "
                f"{item['percent']:>7.2f}%"
            )
    else:
        lines.append("No testable folds were run.")

    lines.extend(
        [
            "",
            "HONESTY CHECK",
            "-" * 72,
            "Hidden expected output passed to solver : NO",
            "Hidden original pair index passed       : NO",
            "Expected-output fallback allowed        : NO",
            "Comparison performed after prediction   : YES",
            "",
            f"Elapsed seconds        : {summary['elapsed_seconds']:.3f}",
            "=" * 72,
        ]
    )

    return "\n".join(lines) + "\n"


def write_reports(
    summary: dict[str, Any],
    task_results: list[dict[str, Any]],
    json_path: str,
    text_path: str,
) -> None:
    json_path = os.path.abspath(json_path)
    text_path = os.path.abspath(text_path)

    json_parent = os.path.dirname(json_path)
    text_parent = os.path.dirname(text_path)

    if json_parent:
        os.makedirs(json_parent, exist_ok=True)
    if text_parent:
        os.makedirs(text_parent, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "summary": summary,
                "tasks": task_results,
            },
            file,
            indent=2,
        )

    with open(text_path, "w", encoding="utf-8") as file:
        file.write(render_summary(summary))


def main() -> int:
    args = parse_args()
    tasks = load_tasks(args.data)

    if args.workers < 1:
        print("--workers must be at least 1", file=sys.stderr)
        return 2

    if args.task is not None:
        tasks = [item for item in tasks if item[0] == args.task]
        if not tasks:
            print(f"Task not found: {args.task}", file=sys.stderr)
            return 2

    if args.limit is not None:
        if args.limit < 1:
            print("--limit must be at least 1", file=sys.stderr)
            return 2
        tasks = tasks[: args.limit]

    fingerprint = solver_fingerprint()
    selected_task_ids = {task_id for task_id, _ in tasks}
    completed_by_id = {}
    previous_elapsed = 0.0

    if args.resume:
        completed_by_id, previous_elapsed = load_compatible_checkpoint(
            os.path.abspath(args.json_report),
            fingerprint,
        )
        completed_by_id = {
            task_id: result
            for task_id, result in completed_by_id.items()
            if task_id in selected_task_ids
        }

    pending_tasks = [
        (task_id, task)
        for task_id, task in tasks
        if task_id not in completed_by_id
    ]

    print("HONEST ARCs4 BENCHMARK")
    print("Hidden outputs and hidden pair indexes will not be passed to the solver.")
    print(f"Tasks selected: {len(tasks)}")
    print(f"Workers: {min(args.workers, max(1, len(pending_tasks)))}")

    if args.resume:
        print(f"Compatible completed tasks reused: {len(completed_by_id)}")

    print()

    started = time.perf_counter()
    task_results = list(completed_by_id.values())

    if args.workers == 1:
        result_iterator = (
            benchmark_task_job(
                task_id=task_id,
                task=task,
                quiet=not args.verbose_router,
            )
            for task_id, task in pending_tasks
        )
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=min(args.workers, max(1, len(pending_tasks)))
        )
        futures = [
            executor.submit(
                benchmark_task_job,
                task_id,
                task,
                not args.verbose_router,
            )
            for task_id, task in pending_tasks
        ]
        result_iterator = (future.result() for future in as_completed(futures))

    try:
        for result in result_iterator:
            task_id = result["task_id"]
            completed_by_id[task_id] = result
            task_results = [
                completed_by_id[selected_id]
                for selected_id, _ in tasks
                if selected_id in completed_by_id
            ]

            if result["status"] == "not_testable":
                score_text = "not testable"
            else:
                score_text = f"{result['honest_exact']}/{result['honest_total']}"

            print(
                f"[{len(task_results):>3}/{len(tasks)}] "
                f"{task_id:<12} honest={score_text:<10} "
                f"replay={result['replay_exact']}/{result['replay_total']}",
                flush=True,
            )

            # Keep a usable checkpoint after every completed task. A matching
            # solver fingerprint lets --resume safely continue it later.
            checkpoint_summary = build_summary(
                data_path=args.data,
                task_results=task_results,
                elapsed_seconds=(
                    previous_elapsed + time.perf_counter() - started
                ),
                fingerprint=fingerprint,
            )
            write_reports(
                summary=checkpoint_summary,
                task_results=task_results,
                json_path=args.json_report,
                text_path=args.text_report,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    elapsed_seconds = previous_elapsed + time.perf_counter() - started
    summary = build_summary(
        data_path=args.data,
        task_results=task_results,
        elapsed_seconds=elapsed_seconds,
        fingerprint=fingerprint,
    )

    write_reports(
        summary=summary,
        task_results=task_results,
        json_path=args.json_report,
        text_path=args.text_report,
    )

    print()
    print(render_summary(summary), end="")
    print(f"Detailed JSON saved to : {os.path.abspath(args.json_report)}")
    print(f"Text summary saved to  : {os.path.abspath(args.text_report)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
