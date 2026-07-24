# audit_motif_layout_attempts.py
import contextlib
import csv
import io
import json
import os
import traceback

from reasoning.motif_layout_rule import solve_pair_motif_layout_rule


DATA_PATH = os.path.join("data", "data.json")
OUT_DIR = os.path.join("data_failures", "motif_layout_audit")

REPORT_JSON = os.path.join(OUT_DIR, "motif_layout_attempts_report.json")
REPORT_CSV = os.path.join(OUT_DIR, "motif_layout_attempts_report.csv")

FULLY_SOLVED_TXT = os.path.join(OUT_DIR, "fully_solved_by_motif_layout.txt")
PARTIAL_SOLVED_TXT = os.path.join(OUT_DIR, "partially_solved_by_motif_layout.txt")
TRIED_TXT = os.path.join(OUT_DIR, "tried_by_motif_layout.txt")
TRIED_NOT_EXACT_TXT = os.path.join(OUT_DIR, "tried_but_not_fully_exact.txt")
ERRORED_TXT = os.path.join(OUT_DIR, "errored_by_motif_layout.txt")


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(str(line) + "\n")


def load_tasks():
    data = load_json(DATA_PATH)

    if isinstance(data, dict) and "train" in data:
        return {"single_task": data}

    if isinstance(data, dict):
        return data

    if isinstance(data, list):
        tasks = {}

        for idx, task in enumerate(data):
            task_id = task.get("id", f"task_{idx:04d}")
            tasks[task_id] = task

        return tasks

    raise ValueError("Unknown data/data.json format")


def run_motif_rule_silent(input_grid, output_grid):
    """
    motif_layout_rule.py prints debug output.
    This suppresses that noise during the audit.
    """
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        result = solve_pair_motif_layout_rule(
            input_grid,
            output_grid,
        )

    debug_text = buffer.getvalue()

    return result, debug_text


def audit_task(task_id, task):
    train_pairs = task.get("train", [])

    pair_results = []

    predicted_count = 0
    exact_count = 0
    total_score = 0
    error_count = 0

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        output_grid = pair["output"]

        try:
            result, debug_text = run_motif_rule_silent(
                input_grid,
                output_grid,
            )

            if result is None:
                pair_results.append({
                    "pair_index": pair_index,
                    "tried": False,
                    "exact": False,
                    "score": None,
                    "strategy": None,
                    "error": None,
                })
                continue

            predicted = result.get("predicted")
            score = result.get("score")
            exact = bool(result.get("exact"))
            strategy = result.get("strategy")

            tried = predicted is not None

            if tried:
                predicted_count += 1

            if exact:
                exact_count += 1

            if isinstance(score, int):
                total_score += score

            pair_results.append({
                "pair_index": pair_index,
                "tried": tried,
                "exact": exact,
                "score": score,
                "strategy": strategy,
                "error": None,
            })

        except Exception as exc:
            error_count += 1

            pair_results.append({
                "pair_index": pair_index,
                "tried": False,
                "exact": False,
                "score": None,
                "strategy": None,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })

    train_count = len(train_pairs)

    tried_any = predicted_count > 0
    tried_all = train_count > 0 and predicted_count == train_count
    fully_solved = train_count > 0 and exact_count == train_count
    partially_solved = exact_count > 0 and exact_count < train_count
    tried_but_not_fully_exact = tried_any and not fully_solved

    return {
        "task_id": task_id,
        "train_count": train_count,
        "predicted_count": predicted_count,
        "exact_count": exact_count,
        "error_count": error_count,
        "total_score": total_score,
        "tried_any": tried_any,
        "tried_all": tried_all,
        "fully_solved": fully_solved,
        "partially_solved": partially_solved,
        "tried_but_not_fully_exact": tried_but_not_fully_exact,
        "pair_results": pair_results,
    }


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = [
        "task_id",
        "train_count",
        "predicted_count",
        "exact_count",
        "error_count",
        "total_score",
        "tried_any",
        "tried_all",
        "fully_solved",
        "partially_solved",
        "tried_but_not_fully_exact",
    ]

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: row.get(key)
                for key in fieldnames
            })


def main():
    print()
    print("=" * 80)
    print("AUDIT motif_layout_rule.py ACROSS ALL TASKS")
    print("=" * 80)

    tasks = load_tasks()

    report_rows = []

    fully_solved_ids = []
    partially_solved_ids = []
    tried_ids = []
    tried_not_exact_ids = []
    errored_ids = []

    for idx, (task_id, task) in enumerate(tasks.items(), start=1):
        result = audit_task(task_id, task)
        report_rows.append(result)

        if result["tried_any"]:
            tried_ids.append(task_id)

        if result["fully_solved"]:
            fully_solved_ids.append(task_id)

        if result["partially_solved"]:
            partially_solved_ids.append(task_id)

        if result["tried_but_not_fully_exact"]:
            tried_not_exact_ids.append(task_id)

        if result["error_count"] > 0:
            errored_ids.append(task_id)

        status = "NO_TRY"

        if result["fully_solved"]:
            status = "FULL"
        elif result["partially_solved"]:
            status = "PART"
        elif result["tried_any"]:
            status = "TRIED"
        elif result["error_count"] > 0:
            status = "ERROR"

        print(
            f"[{status:5}] {task_id} "
            f"pred={result['predicted_count']}/{result['train_count']} "
            f"exact={result['exact_count']}/{result['train_count']} "
            f"errors={result['error_count']} "
            f"score={result['total_score']}"
        )

    report = {
        "source": DATA_PATH,
        "rule": "motif_layout_rule",
        "task_count": len(tasks),
        "fully_solved_count": len(fully_solved_ids),
        "partially_solved_count": len(partially_solved_ids),
        "tried_count": len(tried_ids),
        "tried_not_fully_exact_count": len(tried_not_exact_ids),
        "errored_count": len(errored_ids),
        "fully_solved_ids": fully_solved_ids,
        "partially_solved_ids": partially_solved_ids,
        "tried_ids": tried_ids,
        "tried_not_fully_exact_ids": tried_not_exact_ids,
        "errored_ids": errored_ids,
        "tasks": report_rows,
    }

    save_json(REPORT_JSON, report)
    write_csv(REPORT_CSV, report_rows)

    save_lines(FULLY_SOLVED_TXT, fully_solved_ids)
    save_lines(PARTIAL_SOLVED_TXT, partially_solved_ids)
    save_lines(TRIED_TXT, tried_ids)
    save_lines(TRIED_NOT_EXACT_TXT, tried_not_exact_ids)
    save_lines(ERRORED_TXT, errored_ids)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tasks checked          : {len(tasks)}")
    print(f"Tried / produced prediction  : {len(tried_ids)}")
    print(f"Fully solved                 : {len(fully_solved_ids)}")
    print(f"Partially solved             : {len(partially_solved_ids)}")
    print(f"Tried but not fully exact    : {len(tried_not_exact_ids)}")
    print(f"Errored                      : {len(errored_ids)}")
    print()
    print(f"JSON report : {REPORT_JSON}")
    print(f"CSV report  : {REPORT_CSV}")
    print(f"Full IDs    : {FULLY_SOLVED_TXT}")
    print(f"Partial IDs : {PARTIAL_SOLVED_TXT}")
    print(f"Tried IDs   : {TRIED_TXT}")


if __name__ == "__main__":
    main()